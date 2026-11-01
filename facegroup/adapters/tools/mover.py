"""Moving duplicate copies into the holding folder and back. The only code that changes the user's folders.

Moves stay on the same drive (`<drive>\\FaceGroup removed\\...`), so they are renames: instant, and nothing is
copied or deleted. A move never overwrites anything, in either direction.
"""

from __future__ import annotations

import os
from pathlib import Path


class MoveError(OSError):
    pass


def move_file(src: str, dst: str) -> None:
    """Rename `src` to `dst`, creating folders. Refuses if `dst` exists or `src` changed drive."""
    s, d = Path(src), Path(dst)
    if d.exists():
        raise MoveError(f"something is already at {d}")
    if s.drive.casefold() != d.drive.casefold():
        raise MoveError("the holding folder must be on the same drive")
    d.parent.mkdir(parents=True, exist_ok=True)
    os.rename(s, d)  # not os.replace: never overwrite


def remove_empty_folder(path: str) -> bool:
    """Remove a folder only if it is still empty. Returns whether it was removed."""
    try:
        os.rmdir(path)
        return True
    except OSError:
        return False


def prune_empty_parents(path: str, holding_name: str) -> None:
    """After a restore, remove folders inside the holding folder that became empty, and the holding folder itself
    once nothing is left in it. Never goes above the holding folder."""
    parts = [p for p in Path(path).parents if p.name.casefold() == holding_name.casefold()]
    if not parts:
        return
    top = parts[0]
    p = Path(path).parent
    while p == top or top in p.parents:
        if not remove_empty_folder(str(p)) or p == top:
            break
        p = p.parent
