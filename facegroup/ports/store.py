"""Storage port. Services call these methods; the adapter routes every write through the single DB writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from facegroup.domain.entities import DetectedFace, FileInfo, Job, MediaRecord, Stage


@dataclass(frozen=True)
class Root:
    id: int
    path: Path
    fetch_online_only: bool


@dataclass
class ScanPlan:
    """Everything one scan of one root decided, applied in a single transaction."""

    root_id: int
    new: list[FileInfo] = field(default_factory=list)
    new_online_only: list[FileInfo] = field(default_factory=list)
    changed: list[tuple[int, FileInfo]] = field(default_factory=list)
    reappeared: list[int] = field(default_factory=list)
    hydrated: list[tuple[int, FileInfo]] = field(default_factory=list)
    became_online_only: list[int] = field(default_factory=list)
    moved: list[tuple[int, FileInfo]] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    remove_missing_before: str | None = None  # ISO timestamp; media missing since before this are deleted


@dataclass(frozen=True)
class ImageMeta:
    width: int  # original, after EXIF rotation
    height: int
    taken_at: str | None
    gps: tuple[float, float] | None
    content_hash: str
    working_width: int
    working_height: int


class Store(Protocol):
    def add_root(self, path: Path, fetch_online_only: bool = False) -> Root: ...

    def roots(self) -> list[Root]: ...

    def known_media(self, root_id: int) -> dict[str, MediaRecord]: ...

    def apply_scan(self, plan: ScanPlan) -> int:
        """Apply the plan in one transaction; returns how many long-missing media were removed."""
        ...

    def media_path(self, media_id: int) -> Path: ...

    def release_stale_leases(self, max_attempts: int) -> int:
        """At start-up: jobs still 'running' belong to a dead process. Re-queue them, or mark dead at max attempts."""
        ...

    def lease_jobs(self, stage: Stage, limit: int, lease_seconds: int, max_attempts: int) -> list[Job]: ...

    def create_items(self, job: Job, meta: ImageMeta) -> list[int]:
        """Replace the media's items (a photo has one) and return their ids. Does not finish the job."""
        ...

    def finish_derive(self, job: Job, item_ids: list[int], face_model_id: str) -> None:
        """Mark the derive job done and enqueue the next stages for its items."""
        ...

    def finish_faces(self, job: Job, faces: list[DetectedFace], qualities: list[float],
                     detector_id: str) -> list[int]:
        """Store the faces (reusing overlapping rows so labels survive) and return their ids, in order."""
        ...

    def fail_job(self, job: Job, error: str, permanent: bool, max_attempts: int,
                 backoff_seconds: int) -> None: ...

    def items_with_open_jobs(self, item_ids: list[int]) -> set[int]: ...

    def retry_dead(self) -> int: ...

    def status(self) -> dict[str, dict[str, int]]: ...
