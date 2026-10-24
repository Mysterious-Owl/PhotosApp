"""Domain entities. Pure data: no I/O, no imports from other FaceGroup layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class FileState(StrEnum):
    ACTIVE = "active"
    ONLINE_ONLY = "online_only"
    CHANGED = "changed"
    MISSING = "missing"


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    DONE = "done"
    DEAD = "dead"
    CANCELLED = "cancelled"


class Stage(StrEnum):
    DERIVE = "derive"  # media-level: decode once, working image, thumbnail, items
    FACES = "faces"  # item-level: detect, quality gate, embed
    EMBED = "embed"  # item-level: image embedding for search


@dataclass(frozen=True)
class FileInfo:
    """What a scan learns about a file without reading its contents."""

    rel_path: str
    size: int
    mtime: float
    online_only: bool


@dataclass(frozen=True)
class MediaRecord:
    id: int
    root_id: int
    rel_path: str
    size: int
    mtime: float
    file_state: FileState
    content_hash: str | None
    missing_since: str | None


@dataclass
class DetectedFace:
    """One face found by a FaceAnalyzer, in working-image pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    landmarks: np.ndarray  # (5, 2): eyes, nose, mouth corners
    det_score: float
    embedding: np.ndarray | None = None  # L2-normalised float32
    blur: float = 0.0  # variance of Laplacian on the aligned crop

    @property
    def short_side(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)


@dataclass(frozen=True)
class Job:
    id: int
    media_id: int
    item_id: int | None
    stage: Stage
    model_id: str
    attempts: int


@dataclass
class ScanReport:
    seen: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    moved: int = 0
    online_only: int = 0
    missing: int = 0
    removed: int = 0
    errors: list[str] = field(default_factory=list)
