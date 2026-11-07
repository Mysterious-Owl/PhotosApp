"""LibraryService: library folders, discovery, and change / move / delete detection. Never runs models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from facegroup.adapters.storage.base import iso
from facegroup.domain.entities import FileInfo, FileState, MediaRecord, ScanReport
from facegroup.domain.policies import Change, classify_change, move_candidates
from facegroup.ports.media import FileSource
from facegroup.ports.store import Root, ScanPlan, Store

log = logging.getLogger(__name__)


class LibraryService:
    def __init__(self, store: Store, files: FileSource, missing_grace_days: int = 30,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._store = store
        self._files = files
        self._grace = timedelta(days=missing_grace_days)
        self._now = now

    def add_root(self, path: Path, fetch_online_only: bool = False) -> Root:
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Not a folder: {path}")
        return self._store.add_root(path, fetch_online_only)

    def roots(self) -> list[Root]:
        return self._store.roots()

    def scan(self, root: Root) -> ScanReport:
        report = ScanReport()
        if not root.path.is_dir():
            # A disconnected drive must not mark the whole library missing.
            report.errors.append(f"Folder not available, skipped: {root.path}")
            return report

        known = self._store.known_media(root.id)
        plan = ScanPlan(root_id=root.id)
        seen: set[str] = set()
        new_files = []

        for info in self._files.walk(root.path):
            report.seen += 1
            seen.add(info.rel_path)
            if root.fetch_online_only and info.online_only:
                info = FileInfo(info.rel_path, info.size, info.mtime, online_only=False)
            record = known.get(info.rel_path)
            if record is None:
                new_files.append(info)
                continue
            change = classify_change(record, info)
            if change == Change.UNCHANGED:
                report.unchanged += 1
            elif change == Change.CHANGED:
                plan.changed.append((record.id, info))
                report.changed += 1
            elif change == Change.REAPPEARED:
                plan.reappeared.append(record.id)
                report.unchanged += 1
            elif change == Change.HYDRATED:
                plan.hydrated.append((record.id, info))
                report.new += 1
            elif change == Change.ONLINE_ONLY:
                if record.file_state != FileState.ONLINE_ONLY:
                    plan.became_online_only.append(record.id)
                report.online_only += 1

        gone = [r for path, r in known.items() if path not in seen]
        unmatched = {r.id: r for r in gone}
        for info in new_files:
            moved_from = self._find_move(root, info, list(unmatched.values()))
            if moved_from is not None:
                plan.moved.append((moved_from.id, info))
                del unmatched[moved_from.id]
                report.moved += 1
            elif info.online_only:
                plan.new_online_only.append(info)
                report.online_only += 1
            else:
                plan.new.append(info)
                report.new += 1

        plan.missing = list(unmatched)
        report.missing = sum(1 for r in unmatched.values() if r.file_state != FileState.MISSING)
        plan.remove_missing_before = iso(self._now() - self._grace)
        report.removed = self._store.apply_scan(plan)
        log.info("scan %s: %s", root.path, report)
        return report

    def _find_move(self, root: Root, info: FileInfo, missing: list[MediaRecord]) -> MediaRecord | None:
        candidates = move_candidates(info, missing)
        if not candidates:
            return None
        try:
            digest = self._files.content_hash(root.path / info.rel_path)
        except OSError:
            return None
        return next((c for c in candidates if c.content_hash == digest), None)
