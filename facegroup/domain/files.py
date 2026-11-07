"""Rules for the file tools (duplicate files, HEIC conversion, metadata removal). Pure: paths are strings.

Duplicate files: which copy to keep. The user's own choice for a group wins; then folder preferences ("between
these two folders, keep copies in A"), decided once and applied to every group; then copies inside one folder,
where the plain name beats "IMG_1 (1).jpg" or "IMG_1 - Copy.jpg". Anything else only gets a suggestion, which
is never acted on unless the user says so.

A HEIC photo and a PNG made from it (same name, verified to be the same picture) form a "converted" set: the PNG
is kept, because it is lossless and opens everywhere. Like a suggestion, it is only acted on when the user agrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PureWindowsPath

HEIC_EXTENSIONS = frozenset({".heic", ".heif"})
PHOTO_VIDEO_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif", ".avif",
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".raf", ".rw2",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".mts", ".m2ts", ".wmv", ".webm",
})
METADATA_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif"})
HOLDING_FOLDER = "FaceGroup removed"
CONVERTED_PREFIX = "heic-png:"  # digest prefix of a HEIC + PNG set (not byte-identical, the same picture)
# Among copies that are otherwise equal, the more widely readable format is kept: PNG first, HEIC last.
_FORMAT_RANK = {".png": 0, ".heic": 2, ".heif": 2}

# "IMG_1 (1).jpg", "IMG_1 - Copy.jpg", "IMG_1 - Copy (2).jpg", "IMG_1 copy.jpg" are copies ("IMG_1" is not).
_COPY_NAME = re.compile(r"(\s?\(\d+\)|\s-\s(copy|kopie|copie)(\s\(\d+\))?|\scopy(\s\d+)?)$", re.IGNORECASE)
# Folders that usually hold the spare copy, so a suggestion keeps the copy elsewhere.
_SPARE_FOLDER = re.compile(r"(^|[\\/ _-])(backup|backups|copy|copies|duplicates?|old|temp|tmp|download|downloads|"
                           r"whatsapp|sent|received)([\\/ _-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class Copy:
    path: str
    size: int
    mtime: float


@dataclass
class Decision:
    keep: Copy
    remove: list[Copy]
    reason: str  # "choice" | "rule" | "same_folder" | "converted" | "suggested"

    @property
    def confirmed(self) -> bool:
        """Acted on without asking again. A suggestion, or moving a HEIC original, needs the user's say-so."""
        return self.reason not in ("suggested", "converted")


@dataclass
class FolderPair:
    a: str
    b: str
    groups: int = 0
    extra_bytes: int = 0
    winner: str | None = None  # the folder already preferred (as displayed: a or b), if any
    examples: list[str] = field(default_factory=list)


def folder_of(path: str) -> str:
    return str(PureWindowsPath(path).parent)


def folder_key(folder: str) -> str:
    """Case-insensitive, separator-normalised, as Windows compares folders."""
    return folder.replace("/", "\\").rstrip("\\").casefold()


def is_copy_name(path: str) -> bool:
    return bool(_COPY_NAME.search(PureWindowsPath(path).stem))


def is_converted_set(digest: str) -> bool:
    return digest.startswith(CONVERTED_PREFIX)


def decide(copies: list[Copy], prefs: dict[tuple[str, str], str], choice: str | None = None,
           converted: bool = False) -> Decision:
    """Which copy to keep. `prefs` maps a pair of folder keys (sorted) to the preferred folder key; `choice` is a
    path the user picked for this group; `converted` marks a HEIC + PNG set."""
    copies = sorted(copies, key=lambda c: c.path.casefold())
    if choice is not None:
        keep = next((c for c in copies if folder_key(c.path) == folder_key(choice)), None)
        if keep is not None:
            return Decision(keep, [c for c in copies if c is not keep], "choice")
    if converted:
        keep = min(copies, key=lambda c: (_format_rank(c.path), _suggestion_rank(c)))
        return Decision(keep, [c for c in copies if c is not keep], "converted")
    folders = {folder_key(folder_of(c.path)) for c in copies}
    if len(folders) == 1:
        keep = _best_in_folder(copies)
        return Decision(keep, [c for c in copies if c is not keep], "same_folder")
    winner = _preferred_folder(folders, prefs)
    if winner is not None:
        in_winner = [c for c in copies if folder_key(folder_of(c.path)) == winner]
        keep = _best_in_folder(in_winner)
        return Decision(keep, [c for c in copies if c is not keep], "rule")
    keep = min(copies, key=_suggestion_rank)
    return Decision(keep, [c for c in copies if c is not keep], "suggested")


def _preferred_folder(folders: set[str], prefs: dict[tuple[str, str], str]) -> str | None:
    """The folder preferred over every other folder in the group by the user's pairwise preferences."""
    for f in folders:
        if all(prefs.get(_pair(f, g)) == f for g in folders if g != f):
            return f
    return None


def _format_rank(path: str) -> int:
    return _FORMAT_RANK.get(PureWindowsPath(path).suffix.lower(), 1)


def _best_in_folder(copies: list[Copy]) -> Copy:
    # The plain name, then PNG over HEIC, then the oldest file, then the shortest name.
    return min(copies, key=lambda c: (is_copy_name(c.path), _format_rank(c.path), c.mtime, len(c.path),
                                      c.path.casefold()))


def _suggestion_rank(c: Copy) -> tuple:
    return (bool(_SPARE_FOLDER.search(folder_of(c.path))), is_copy_name(c.path), _format_rank(c.path), c.mtime,
            len(c.path))


def converted_candidates(heics: list[Copy], pngs: list[Copy], per_heic: int = 5) -> list[tuple[Copy, Copy]]:
    """(HEIC, PNG) pairs with the same file name, to be checked for being the same picture. A PNG in the same
    folder or in "<folder> converted" is tried first."""
    by_stem: dict[str, list[Copy]] = {}
    for p in pngs:
        by_stem.setdefault(PureWindowsPath(p.path).stem.casefold(), []).append(p)
    out = []
    for h in heics:
        hp = PureWindowsPath(h.path)
        near = {folder_key(str(hp.parent)), folder_key(f"{hp.parent} converted")}
        found = sorted(by_stem.get(hp.stem.casefold(), []), key=lambda p: folder_key(folder_of(p.path)) not in near)
        out += [(h, p) for p in found[:per_heic]]
    return out


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def pair_key(a: str, b: str) -> tuple[str, str]:
    """The key `prefs` uses for two folders (any case or separator)."""
    return _pair(folder_key(a), folder_key(b))


def folder_pairs(groups: list[list[Copy]], prefs: dict[tuple[str, str], str], limit_examples: int = 3) -> list[FolderPair]:
    """Folder pairs that share duplicates, most space first: deciding one pair settles all its groups."""
    pairs: dict[tuple[str, str], FolderPair] = {}
    for copies in groups:
        by_folder: dict[str, str] = {}
        for c in copies:
            by_folder.setdefault(folder_key(folder_of(c.path)), folder_of(c.path))
        keys = sorted(by_folder)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                p = pairs.get((a, b))
                if p is None:
                    keep = prefs.get((a, b))
                    p = pairs[(a, b)] = FolderPair(by_folder[a], by_folder[b], winner=by_folder.get(keep) if keep else None)
                p.groups += 1
                p.extra_bytes += min(c.size for c in copies)
                if len(p.examples) < limit_examples:
                    p.examples.append(PureWindowsPath(copies[0].path).name)
    return sorted(pairs.values(), key=lambda p: (-p.extra_bytes, p.a.casefold(), p.b.casefold()))


def holding_path(original: str, taken: set[str] | None = None) -> str:
    """Where a removed copy goes: `<drive>\\FaceGroup removed\\<path below the drive>`, on the same drive so the
    move is a rename. A clash with an earlier removal gets " (2)", " (3)"... in `taken` (casefolded paths)."""
    p = PureWindowsPath(original)
    base = PureWindowsPath(p.anchor) / HOLDING_FOLDER / p.relative_to(p.anchor)
    candidate, n = base, 2
    while taken is not None and str(candidate).casefold() in taken:
        candidate = base.with_name(f"{base.stem} ({n}){base.suffix}")
        n += 1
    return str(candidate)


def converted_name(name: str, fmt: str) -> str:
    """'IMG_1.HEIC' -> 'IMG_1.png' (HEIC files only; others keep their name)."""
    p = PureWindowsPath(name)
    return p.stem + "." + fmt if p.suffix.lower() in HEIC_EXTENSIONS else p.name


def sibling_folder(folder: str, suffix: str) -> str:
    """'D:\\Trip' -> 'D:\\Trip converted'. A drive root gets a folder inside it instead."""
    p = PureWindowsPath(folder)
    if str(p) == p.anchor:
        return str(p / f"FaceGroup {suffix}")
    return str(p.with_name(f"{p.name} {suffix}"))


def output_folder(output: str, src: str, suffix: str) -> str:
    """The output folder the user gave (quotes from "Copy as path" removed), else '<src> <suffix>' next to it."""
    output = output.strip()
    return output.strip('"') if output else sibling_folder(src, suffix)
