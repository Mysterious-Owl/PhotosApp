"""Walking any folder the user picks, for the file tools. Directory metadata only: no file is opened, so OneDrive
online-only files are never downloaded."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from facegroup.adapters.media.files import hash_file, is_online_only
from facegroup.domain.files import HOLDING_FOLDER

# Never walked: Windows system folders, our own holding folder, and developer/cache trees full of tiny files.
SKIP_DIRS = frozenset(name.casefold() for name in (
    "$recycle.bin", "system volume information", HOLDING_FOLDER, "windows", "program files", "program files (x86)",
    "programdata", "appdata", ".git", "node_modules", "__pycache__", ".venv", "venv",
))


@dataclass(frozen=True)
class FoundFile:
    path: str
    size: int
    mtime: float
    online_only: bool


def walk(roots: list[Path], extensions: frozenset[str] | None, should_stop: Callable[[], bool] = lambda: False,
         errors: list[str] | None = None) -> Iterator[FoundFile]:
    """Every regular file under `roots` (only `extensions`, if given). Links and junctions are not followed."""
    stack = list(roots)
    while stack and not should_stop():
        folder = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError as exc:
            if errors is not None:
                errors.append(f"{folder}: {exc.strerror or exc}")
            continue
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                if entry.name.casefold() not in SKIP_DIRS and not _is_link(st):
                    stack.append(Path(entry.path))
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if extensions is not None and Path(entry.name).suffix.lower() not in extensions:
                continue
            yield FoundFile(entry.path, st.st_size, st.st_mtime, is_online_only(st))


def empty_folders(roots: list[Path], should_stop: Callable[[], bool] = lambda: False) -> list[str]:
    """Folders with nothing in them, including ones that hold only empty folders. The roots themselves are kept."""
    out: list[str] = []

    def visit(folder: Path) -> bool:
        if should_stop():
            return False
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return False
        empty = True
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                empty = False
                continue
            if stat.S_ISDIR(st.st_mode) and not _is_link(st) and entry.name.casefold() not in SKIP_DIRS:
                if visit(Path(entry.path)):
                    out.append(entry.path)
                else:
                    empty = False
            else:
                empty = False
        return empty

    for root in roots:
        visit(root)
    return sorted(out, key=str.casefold)


def quick_hash(path: str, block: int = 64 * 1024) -> str:
    """First and last 64 KB plus the size: cheap, and different files almost never collide. Confirmed by full_hash."""
    h = hashlib.blake2b(digest_size=16)
    size = os.path.getsize(path)
    h.update(size.to_bytes(8, "little"))
    with open(path, "rb") as f:
        h.update(f.read(block))
        if size > 2 * block:
            f.seek(-block, os.SEEK_END)
            h.update(f.read(block))
    return h.hexdigest()


def full_hash(path: str) -> str:
    return hash_file(Path(path))


_LINK_TAGS = frozenset({getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
                        getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C)})


def _is_link(st: os.stat_result) -> bool:
    # Junctions and symlinks are not followed (they can loop). OneDrive placeholders are reparse points too, but
    # with other tags, so they are still walked.
    return stat.S_ISLNK(st.st_mode) or getattr(st, "st_reparse_tag", 0) in _LINK_TAGS
