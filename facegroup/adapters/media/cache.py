"""Working images and thumbnails on local disk. Paths are written as bytes, so non-ASCII user folders work."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DiskImageCache:
    def __init__(self, root: Path, thumbnail_size: int = 320) -> None:
        self._work = root / "work"
        self._thumbs = root / "thumbs"
        self._thumb_size = thumbnail_size

    @staticmethod
    def _shard(base: Path, key: int) -> Path:
        return base / f"{key // 1000:05d}" / f"{key}.jpg"

    def working_path(self, item_id: int) -> Path:
        return self._shard(self._work, item_id)

    def thumbnail_path(self, media_id: int) -> Path:
        return self._shard(self._thumbs, media_id)

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
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)  # atomic: a crash never leaves a half-written image
