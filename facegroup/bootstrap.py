"""The only place that knows which adapter implements which port."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

from facegroup.adapters.instance_lock import InstanceLock
from facegroup.adapters.media.cache import DiskImageCache
from facegroup.adapters.media.decoder import PillowDecoder
from facegroup.adapters.media.files import LocalFileSource
from facegroup.adapters.storage.db import assert_not_synced, connect, migrate
from facegroup.adapters.storage.sqlite_store import SqliteStore
from facegroup.adapters.storage.writer import DbWriter
from facegroup.config import Settings
from facegroup.domain.policies import QualityGate
from facegroup.ports.face import FaceAnalyzer
from facegroup.services.indexing import IndexingService, IndexingSettings, Progress
from facegroup.services.library import LibraryService


@dataclass
class App:
    settings: Settings
    store: SqliteStore
    library: LibraryService
    indexing: IndexingService
    _writer: DbWriter
    _lock: InstanceLock

    def close(self) -> None:
        self.store.close()
        self._writer.close()
        self._lock.release()


def face_analyzer_factory(settings: Settings) -> tuple[Callable[[], FaceAnalyzer], str]:
    if settings.face_model != "buffalo_l":
        raise ValueError(f"Unknown face_model {settings.face_model!r}")
    from facegroup.adapters.models.buffalo_onnx import BuffaloL, ensure_buffalo_l

    def make() -> FaceAnalyzer:
        return BuffaloL(ensure_buffalo_l(settings.models_dir), threads=settings.ort_threads, device=settings.device)

    return make, BuffaloL.model_id


def setup_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(settings.log_dir / "facegroup.log", maxBytes=5_000_000, backupCount=3,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)


def build(settings: Settings, *, analyzer_factory: Callable[[], FaceAnalyzer] | None = None,
          face_model_id: str | None = None, progress: Progress | None = None,
          should_stop: Callable[[], bool] = lambda: False) -> App:
    """Wire adapters into services. Tests pass a fake analyzer_factory + face_model_id."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    assert_not_synced(settings.data_dir)
    lock = InstanceLock(settings.data_dir / "facegroup.lock")
    lock.acquire()
    try:
        conn = connect(settings.db_path)
        migrate(conn)
        conn.close()
        writer = DbWriter(settings.db_path)
    except BaseException:
        lock.release()
        raise

    store = SqliteStore(settings.db_path, writer)
    if analyzer_factory is None:
        analyzer_factory, face_model_id = face_analyzer_factory(settings)
    assert face_model_id is not None

    indexing = IndexingService(
        store=store,
        decoder=PillowDecoder(settings.working_max_side, settings.max_image_pixels),
        cache=DiskImageCache(settings.cache_dir, settings.thumbnail_size),
        face_analyzer_factory=analyzer_factory,
        face_model_id=face_model_id,
        gate=QualityGate(settings.min_face_px, settings.min_det_score),
        settings=IndexingSettings(
            batch_size=settings.batch_size,
            decode_workers=settings.decode_workers,
            lease_seconds=settings.lease_seconds,
            max_attempts=settings.max_attempts,
            backoff_seconds=settings.backoff_seconds,
            keep_working_images=settings.keep_working_images,
        ),
        progress=progress,
        should_stop=should_stop,
    )
    library = LibraryService(store, LocalFileSource(), settings.missing_grace_days)
    return App(settings, store, library, indexing, writer, lock)
