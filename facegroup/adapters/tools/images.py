"""Image file operations for the tools: HEIC conversion and metadata removal. Originals are only ever read.

Every output is written to `<name>.part` first and renamed, so a stopped run never leaves a half-written file.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_DATETIME = 0x0132
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
_SAVE_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF",
                ".heic": "HEIF", ".heif": "HEIF"}


def _atomic(dst: Path, write) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        write(tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


# -- HEIC conversion -----------------------------------------------------------------------------------------

def convert_heic(src: Path, dst: Path, fmt: str, jpeg_quality: int = 92, png_compress: int = 3) -> None:
    """HEIC/HEIF to PNG (lossless) or JPEG. Keeps the EXIF (date taken, camera, GPS) and colour profile."""
    with Image.open(src) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        extra = {k: v for k, v in (("exif", exif), ("icc_profile", icc)) if v}
        if fmt == "jpg":
            rgb = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            _atomic(dst, lambda p: rgb.save(p, "JPEG", quality=jpeg_quality, subsampling=0, **extra))
        else:
            _atomic(dst, lambda p: im.save(p, "PNG", compress_level=png_compress, **extra))


def copy_file(src: Path, dst: Path) -> None:
    _atomic(dst, lambda p: shutil.copy2(src, p))


def same_picture(a: Path, b: Path, tolerance: float = 4.0) -> bool:
    """Whether two files hold the same picture in different formats (e.g. a HEIC and the PNG made from it): equal
    dimensions, and nearly equal pixels at 64x64 (mean difference under `tolerance` of 255; HEIC decoding is not
    bit-exact). Dimensions come from the headers, so most non-matches are decided without decoding."""
    import numpy as np

    with Image.open(a) as ia, Image.open(b) as ib:
        if ia.size != ib.size:
            return False
        small = [np.asarray(im.convert("RGB").resize((64, 64), Image.Resampling.BOX), np.int16) for im in (ia, ib)]
    return float(np.abs(small[0] - small[1]).mean()) < tolerance


# -- metadata --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MetadataReport:
    exif_tags: int
    gps: bool
    date_taken: bool
    xmp: bool
    iptc: bool
    icc: bool
    comments: int  # other text: PNG text chunks, JPEG comments, HEIF metadata blocks

    @property
    def clean(self) -> bool:
        return not (self.exif_tags or self.gps or self.xmp or self.iptc or self.comments)


def inspect(path: Path) -> MetadataReport:
    """What metadata a file carries. Values are never returned, only whether each kind is present."""
    with Image.open(path) as im:
        exif = im.getexif()
        gps = bool(exif.get_ifd(GPS_IFD)) if GPS_IFD in exif else False
        sub = exif.get_ifd(EXIF_IFD) if EXIF_IFD in exif else {}
        info = im.info
        text = getattr(im, "text", None) or {}  # PNG text chunks
        return MetadataReport(
            exif_tags=len(exif) + len(sub),
            gps=gps,
            date_taken=EXIF_DATETIME_ORIGINAL in sub or EXIF_DATETIME in exif,
            xmp=bool(info.get("xmp") or info.get("XML:com.adobe.xmp")),
            iptc=bool(info.get("photoshop")),
            icc="icc_profile" in info,
            # PNG text, JPEG/GIF comments, and other HEIF metadata blocks (format fields like bit depth don't count)
            comments=len(text) + bool(info.get("comment")) + len(info.get("metadata") or []),
        )


def strip_metadata(src: Path, dst: Path, keep_icc: bool = True, keep_date: bool = False, quality: int = 95) -> None:
    """Write a copy with no EXIF, GPS, XMP, IPTC or comments. The picture is turned upright first (the orientation
    tag goes too). The colour profile is kept by default, so colours look the same; `keep_date` keeps only the
    date taken."""
    with Image.open(src) as im:
        date = None
        if keep_date:
            exif = im.getexif()
            date = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) if EXIF_IFD in exif else None
            date = date or exif.get(EXIF_DATETIME)
        icc = im.info.get("icc_profile") if keep_icc else None
        upright = ImageOps.exif_transpose(im)
        # A new image from the pixels alone: nothing in .info or the EXIF cache can carry over.
        clean = Image.frombytes(upright.mode, upright.size, upright.tobytes())
        if upright.mode == "P":
            clean.putpalette(upright.getpalette())
        fmt = _SAVE_FORMAT.get(src.suffix.lower(), "PNG")
        opts: dict = {}
        if icc:
            opts["icc_profile"] = icc
        if date:
            new_exif = Image.Exif()
            new_exif.get_ifd(EXIF_IFD)[EXIF_DATETIME_ORIGINAL] = date
            opts["exif"] = new_exif.tobytes()
        if fmt == "JPEG":
            if clean.mode not in ("RGB", "L"):
                clean = clean.convert("RGB")
            opts.update(quality=quality, subsampling=0, optimize=True)
        elif fmt in ("WEBP", "HEIF"):
            opts.update(quality=quality)
        elif fmt == "PNG":
            opts.update(optimize=True)
        _atomic(dst, lambda p: clean.save(p, fmt, **opts))
