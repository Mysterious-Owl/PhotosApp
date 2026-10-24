from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from facegroup.domain.entities import FileInfo


class PermanentMediaError(Exception):
    """The file can never be decoded (corrupt, unsupported, too large). Do not retry."""


@dataclass(frozen=True)
class DecodedImage:
    rgb: np.ndarray  # working image: H x W x 3 uint8, EXIF-rotated, longest side capped
    width: int  # original size after EXIF rotation
    height: int
    content_hash: str  # of the file bytes, computed while reading them (used for move detection)
    taken_at: str | None = None  # ISO 8601 from EXIF, if present
    gps: tuple[float, float] | None = None


class FileSource(Protocol):
    def walk(self, root: Path) -> Iterator[FileInfo]:
        """Yield supported media under root without reading file contents."""
        ...

    def content_hash(self, path: Path) -> str: ...


class MediaDecoder(Protocol):
    def decode(self, path: Path) -> DecodedImage:
        """Raise PermanentMediaError for files that will never decode; OSError for transient failures."""
        ...


class ImageCache(Protocol):
    def write_working(self, item_id: int, rgb: np.ndarray) -> None: ...

    def read_working(self, item_id: int) -> np.ndarray: ...

    def delete_working(self, item_id: int) -> None: ...

    def write_thumbnail(self, media_id: int, rgb: np.ndarray) -> None: ...

    def write_face_crop(self, face_id: int, rgb: np.ndarray, bbox: tuple[float, float, float, float]) -> None: ...

    def delete_face_crop(self, face_id: int) -> None: ...
