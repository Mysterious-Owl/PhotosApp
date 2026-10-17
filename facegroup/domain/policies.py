"""Domain policies: small pure functions that encode the rules from docs/ARCHITECTURE.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from facegroup.domain.entities import DetectedFace, FileInfo, FileState, MediaRecord


class Change(StrEnum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    REAPPEARED = "reappeared"
    ONLINE_ONLY = "online_only"
    HYDRATED = "hydrated"  # was online-only, now local


def classify_change(known: MediaRecord, seen: FileInfo) -> Change:
    """Decide what happened to a file we already know, from size/mtime/attributes only."""
    if seen.online_only:
        return Change.ONLINE_ONLY
    if known.file_state == FileState.ONLINE_ONLY:
        return Change.HYDRATED
    same = known.size == seen.size and abs(known.mtime - seen.mtime) < 1e-6
    if known.file_state == FileState.MISSING:
        return Change.REAPPEARED if same else Change.CHANGED
    # A CHANGED record already stores the new size/mtime, so "same" means its re-index is just outstanding.
    return Change.UNCHANGED if same else Change.CHANGED


def move_candidates(new_file: FileInfo, missing: list[MediaRecord]) -> list[MediaRecord]:
    """Missing records that could be this new file, moved. Only those with a stored hash can be confirmed,
    and an online-only new file is never hashed (that would download it)."""
    if new_file.online_only:
        return []
    return [m for m in missing if m.size == new_file.size and m.content_hash]


Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(old: list[Box], new: list[Box], min_iou: float = 0.5) -> dict[int, int]:
    """Greedy one-to-one match of new boxes to old boxes by IoU. Returns {new_index: old_index}.
    Used so a re-detection keeps the face rows (and their labels) it overlaps."""
    pairs = sorted(
        ((iou(o, n), ni, oi) for oi, o in enumerate(old) for ni, n in enumerate(new)),
        reverse=True,
    )
    matched: dict[int, int] = {}
    used_old: set[int] = set()
    for score, ni, oi in pairs:
        if score < min_iou:
            break
        if ni not in matched and oi not in used_old:
            matched[ni] = oi
            used_old.add(oi)
    return matched


@dataclass(frozen=True)
class QualityGate:
    min_face_px: float = 40.0
    min_det_score: float = 0.5

    def accepts(self, face: DetectedFace) -> bool:
        return face.short_side >= self.min_face_px and face.det_score >= self.min_det_score

    @staticmethod
    def score(face: DetectedFace) -> float:
        """0..1: higher is a better reference face. Size and sharpness saturate at typical good values."""
        size = min(1.0, face.short_side / 112.0)
        sharp = min(1.0, face.blur / 100.0) if face.blur > 0 else 0.5
        return round(face.det_score * size * sharp, 4)
