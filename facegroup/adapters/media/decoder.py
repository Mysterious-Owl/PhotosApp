"""Pillow decoder: reads bytes (so any Windows path works), applies EXIF rotation, caps size, guards bombs."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError

from facegroup.adapters.media.files import hash_bytes
from facegroup.ports.media import DecodedImage, PermanentMediaError

pillow_heif.register_heif_opener()

_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
_DATETIME_ORIGINAL = 36867
_DATETIME = 306


class PillowDecoder:
    def __init__(self, max_side: int = 2048, max_pixels: int = 250_000_000) -> None:
        self._max_side = max_side
        Image.MAX_IMAGE_PIXELS = max_pixels  # larger images raise DecompressionBombError

    def decode(self, path: Path) -> DecodedImage:
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise PermanentMediaError(f"file not found: {path}") from exc
        # Other OSErrors (locked file, drive busy) propagate as transient.
        try:
            with Image.open(io.BytesIO(data)) as im:
                exif = im.getexif()
                orientation = exif.get(0x0112, 1)
                w, h = im.size
                if orientation in (5, 6, 7, 8):
                    w, h = h, w
                im.draft("RGB", (self._max_side, self._max_side))  # JPEG: decode at reduced scale, much faster
                out = ImageOps.exif_transpose(im)
                out = out.convert("RGB")
                out.thumbnail((self._max_side, self._max_side), Image.Resampling.LANCZOS)
                rgb = np.asarray(out, dtype=np.uint8).copy()
        except (UnidentifiedImageError, Image.DecompressionBombError, SyntaxError, ValueError, OSError) as exc:
            # The bytes were read fine, so a decode failure will not fix itself on retry.
            raise PermanentMediaError(f"{type(exc).__name__}: {exc}") from exc
        return DecodedImage(
            rgb=rgb,
            width=w,
            height=h,
            content_hash=hash_bytes(data),
            taken_at=_taken_at(exif),
            gps=_gps(exif),
        )


def _taken_at(exif: Image.Exif) -> str | None:
    raw = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
    if not isinstance(raw, str) or len(raw) < 19:
        return None
    date, _, time = raw.strip().partition(" ")
    parts = date.split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts) or parts[0] == "0000":
        return None
    return f"{parts[0]}-{parts[1]}-{parts[2]}T{time[:8]}"


def _gps(exif: Image.Exif) -> tuple[float, float] | None:
    try:
        g = exif.get_ifd(_GPS_IFD)
        if not g or 2 not in g or 4 not in g:
            return None
        lat = _dms(g[2]) * (-1 if g.get(1) == "S" else 1)
        lon = _dms(g[4]) * (-1 if g.get(3) == "W" else 1)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            return None
        return round(lat, 7), round(lon, 7)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600
