"""Walks library folders using only directory metadata, so OneDrive online-only files are never downloaded."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

from facegroup.domain.entities import FileInfo

PHOTO_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"})
SKIP_DIRS = frozenset({"$recycle.bin", "system volume information"})

# Windows attributes that mean "contents live in the cloud; reading them triggers a download".
FILE_ATTRIBUTE_OFFLINE = 0x00001000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_CLOUD_ONLY = FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS


def is_online_only(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & _CLOUD_ONLY)


class LocalFileSource:
    def __init__(self, extensions: frozenset[str] = PHOTO_EXTENSIONS) -> None:
        self._extensions = extensions

    def walk(self, root: Path) -> Iterator[FileInfo]:
        stack = [root]
        while stack:
            folder = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue  # unreadable folder: skip, the rest of the library still scans
            for entry in entries:
                name = entry.name
                try:
                    # On Windows, scandir's stat comes from the directory listing: no file is opened.
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode):
                    if not name.startswith(".") and name.lower() not in SKIP_DIRS:
                        stack.append(Path(entry.path))
                    continue
                if not stat.S_ISREG(st.st_mode) or Path(name).suffix.lower() not in self._extensions:
                    continue
                rel = Path(entry.path).relative_to(root).as_posix()
                yield FileInfo(rel_path=rel, size=st.st_size, mtime=st.st_mtime, online_only=is_online_only(st))

    def content_hash(self, path: Path) -> str:
        return hash_file(path)


def hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=20).hexdigest()


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
