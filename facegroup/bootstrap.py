"""The only place that knows which adapter implements which port."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

from facegroup.adapters.clustering import HdbscanClusterer
from facegroup.adapters.instance_lock import InstanceLock
from facegroup.adapters.media.cache import DiskImageCache
from facegroup.adapters.media.decoder import PillowDecoder
from facegroup.adapters.media.files import LocalFileSource
from facegroup.adapters.storage.db import assert_not_synced, connect, migrate
from facegroup.adapters.storage.people_store import SqlitePeopleStore
from facegroup.adapters.storage.search_store import SqliteSearchStore
from facegroup.adapters.storage.sqlite_store import SqliteStore
from facegroup.adapters.storage.writer import DbWriter
from facegroup.config import Settings
from facegroup.domain.people import AssignPolicy
from facegroup.domain.policies import QualityGate
from facegroup.ports.face import FaceAnalyzer
from facegroup.services.indexing import IndexingService, IndexingSettings, Progress
from facegroup.services.library import LibraryService
from facegroup.services.media import MediaService
from facegroup.services.people import PeopleService, SuggestPolicy
from facegroup.services.search import SearchService
from facegroup.ports.search import Embedder


@dataclass
class App:
    settings: Settings
    store: SqliteStore
    people_store: SqlitePeopleStore
    cache: DiskImageCache
    decoder: PillowDecoder
    library: LibraryService
    indexing: IndexingService
    people: PeopleService
    media: MediaService
    search: SearchService
    search_store: SqliteSearchStore
    _writer: DbWriter
    _lock: InstanceLock

    def close(self) -> None:
        self.search.close()
        self.store.close()
        self.people_store.close()
        self.search_store.close()
        self._writer.close()
        self._lock.release()


def face_analyzer_factory(settings: Settings) -> tuple[Callable[[], FaceAnalyzer], str]:
    if settings.face_model != "buffalo_l":
        raise ValueError(f"Unknown face_model {settings.face_model!r}")
    from facegroup.adapters.models.buffalo_onnx import BuffaloL, ensure_buffalo_l

    def make() -> FaceAnalyzer:
        return BuffaloL(ensure_buffalo_l(settings.models_dir), threads=settings.ort_threads, device=settings.device)

    return make, BuffaloL.model_id


def embedder_factory(settings: Settings) -> tuple[Callable[[], Embedder] | None, str | None]:
    if settings.search_model == "none":
        return None, None
    from facegroup.adapters.models.siglip2_onnx import SigLip2

    if settings.search_model != "siglip2-base":
        raise ValueError(f"Unknown search_model {settings.search_model!r}")

    def make() -> Embedder:
        return SigLip2(settings.models_dir, threads=settings.ort_threads, device=settings.device)

    return make, SigLip2.model_id


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
          should_stop: Callable[[], bool] = lambda: False,
          embedder: Callable[[], Embedder] | None = None, embed_model_id: str | None = None) -> App:
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
    people_store = SqlitePeopleStore(settings.db_path, writer)
    cache = DiskImageCache(settings.cache_dir, settings.thumbnail_size)
    decoder = PillowDecoder(settings.working_max_side, settings.max_image_pixels)
    if analyzer_factory is None:
        analyzer_factory, face_model_id = face_analyzer_factory(settings)
    assert face_model_id is not None
    if embedder is None and embed_model_id is None:
        embedder, embed_model_id = embedder_factory(settings)

    indexing = IndexingService(
        store=store,
        decoder=decoder,
        cache=cache,
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
        suppressed=people_store.suppressed_vectors,
        embedder_factory=embedder,
        embed_model_id=embed_model_id,
    )
    library = LibraryService(store, LocalFileSource(), settings.missing_grace_days)

    def remove_crops(face_ids: list[int]) -> None:
        for fid in face_ids:
            cache.delete_face_crop(fid)

    people = PeopleService(
        people_store,
        HdbscanClusterer(min_cluster_size=settings.suggest_min_group),
        face_model_id,
        AssignPolicy(settings.assign_k, settings.assign_threshold, settings.assign_min_votes),
        SuggestPolicy(settings.suggest_min_quality, settings.suggest_member_similarity,
                      settings.suggest_merge_similarity, settings.suggest_min_group),
        on_forget=remove_crops,
    )
    media = MediaService(store, people_store, decoder, cache)
    search_store = SqliteSearchStore(settings.db_path, writer)
    search = SearchService(search_store, embedder or (lambda: None), embed_model_id or "")
    return App(settings, store, people_store, cache, decoder, library, indexing, people, media, search,
               search_store, writer, lock)
