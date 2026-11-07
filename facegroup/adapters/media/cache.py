"""Working images and thumbnails on local disk. Paths are written as bytes, so non-ASCII user folders work."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np


class DiskImageCache:
    def __init__(self, root: Path, thumbnail_size: int = 320) -> None:
        self._work = root / "work"
        self._thumbs = root / "thumbs"
        self._faces = root / "faces"
        self._thumb_size = thumbnail_size

    @staticmethod
    def _shard(base: Path, key: int) -> Path:
        return base / f"{key // 1000:05d}" / f"{key}.jpg"

    def working_path(self, item_id: int) -> Path:
        return self._shard(self._work, item_id)

    def thumbnail_path(self, media_id: int) -> Path:
        return self._shard(self._thumbs, media_id)

    def face_crop_path(self, face_id: int) -> Path:
        return self._shard(self._faces, face_id)

    def write_face_crop(self, face_id: int, rgb: np.ndarray, bbox: tuple[float, float, float, float],
                        size: int = 160, pad: float = 0.3) -> None:
        """A square close-up around the face, for the People screens."""
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * (0.5 + pad)
        left, top, right, bottom = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
        crop = rgb[max(0, top):min(h, bottom), max(0, left):min(w, right)]
        if crop.size == 0:
            return
        # Keep the face centred when the square runs off the image edge.
        crop = cv2.copyMakeBorder(crop, max(0, -top), max(0, bottom - h), max(0, -left), max(0, right - w),
                                  cv2.BORDER_CONSTANT, value=(24, 24, 24))
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        _write_jpeg(self.face_crop_path(face_id), crop, quality=88)

    def delete_face_crop(self, face_id: int) -> None:
        self.face_crop_path(face_id).unlink(missing_ok=True)

    def write_working(self, item_id: int, rgb: np.ndarray) -> None:
        _write_jpeg(self.working_path(item_id), rgb, quality=92)

    def read_working(self, item_id: int) -> np.ndarray:
        data = np.frombuffer(self.working_path(item_id).read_bytes(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"unreadable working image for item {item_id}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def delete_working(self, item_id: int) -> None:
        self.working_path(item_id).unlink(missing_ok=True)

    def write_thumbnail(self, media_id: int, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        scale = self._thumb_size / max(h, w)
        if scale < 1:
            rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        _write_jpeg(self.thumbnail_path(media_id), rgb, quality=85)


def _write_jpeg(path: Path, rgb: np.ndarray, quality: int) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per thread: the indexer and a web request can write the same thumbnail at the same time.
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)  # atomic: a crash never leaves a half-written image
