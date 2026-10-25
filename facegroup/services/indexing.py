"""IndexingService: runs the stage DAG in passes (derive, then faces), one model at a time.

It stores faces but never decides who they are; that is PeopleService's job (M2).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from facegroup.domain.entities import Job, Stage
from facegroup.domain.policies import QualityGate
from facegroup.ports.face import FaceAnalyzer
from facegroup.ports.media import ImageCache, MediaDecoder, PermanentMediaError, TextReader
from facegroup.ports.search import Embedder
from facegroup.ports.store import ImageMeta, Store

log = logging.getLogger(__name__)

Progress = Callable[[Stage, int, int], None]  # stage, done this run, failed this run


@dataclass(frozen=True)
class IndexingSettings:
    batch_size: int = 16
    decode_workers: int = 3
    lease_seconds: int = 900
    max_attempts: int = 3
    backoff_seconds: int = 30
    keep_working_images: bool = False


@dataclass
class StageReport:
    stage: Stage
    done: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class IndexingService:
    def __init__(self, store: Store, decoder: MediaDecoder, cache: ImageCache,
                 face_analyzer_factory: Callable[[], FaceAnalyzer], face_model_id: str,
                 gate: QualityGate, settings: IndexingSettings = IndexingSettings(),
                 progress: Progress | None = None, should_stop: Callable[[], bool] = lambda: False,
                 suppressed: Callable[[str], np.ndarray] = lambda model_id: np.zeros((0, 512), np.float32),
                 suppress_similarity: float = 0.5,
                 embedder_factory: Callable[[], Embedder] | None = None, embed_model_id: str | None = None,
                 phash: Callable[[np.ndarray], int] | None = None,
                 reader_factory: Callable[[], TextReader] | None = None, ocr_model_id: str | None = None,
                 ocr_all_photos: bool = False) -> None:
        self._store = store
        self._decoder = decoder
        self._cache = cache
        self._make_analyzer = face_analyzer_factory
        self._face_model_id = face_model_id
        self._gate = gate
        self._s = settings
        self._progress = progress or (lambda *a: None)
        self._should_stop = should_stop
        self._suppressed_source = suppressed
        self._suppress_similarity = suppress_similarity
        self._suppressed = np.zeros((0, 512), np.float32)
        self._phash = phash
        self._make_reader = reader_factory
        self._ocr_model_id = ocr_model_id if reader_factory else None
        self._ocr_all = ocr_all_photos
        self._make_embedder = embedder_factory
        self._embed_model_id = embed_model_id if embedder_factory else None

    def _next_stages(self) -> list[tuple[Stage, str]]:
        stages = [(Stage.FACES, self._face_model_id)]
        if self._embed_model_id:
            stages.append((Stage.EMBED, self._embed_model_id))
        return stages

    def recover(self) -> int:
        """Call once at start-up, while holding the instance lock. Also queues work for any stage a model
        has not processed yet (photos indexed before search existed, or after a model change)."""
        released = self._store.release_stale_leases(self._s.max_attempts)
        for stage, model_id in self._next_stages():
            self._store.enqueue_missing(stage, model_id)
        return released

    def run_all(self) -> list[StageReport]:
        """Every pass in order. The one place that knows the stage order (CLI and scheduler both use it)."""
        return [self.run_derive(), self.run_faces(), self.run_embed()]

    # -- derive: decode once, working image + thumbnail, items ---------------------------------------

    def run_derive(self) -> StageReport:
        report = StageReport(Stage.DERIVE)
        with ThreadPoolExecutor(self._s.decode_workers, thread_name_prefix="decode") as pool:
            while not self._should_stop():
                jobs = self._store.lease_jobs(Stage.DERIVE, self._s.decode_workers * 2,
                                              self._s.lease_seconds, self._s.max_attempts)
                if not jobs:
                    break
                for ok in pool.map(lambda j: self._guarded(j, self._derive_one, report), jobs):
                    self._count(report, ok)
        return report

    def _derive_one(self, job: Job) -> None:
        decoded = self._decoder.decode(self._store.media_path(job.media_id))
        h, w = decoded.rgb.shape[:2]
        meta = ImageMeta(decoded.width, decoded.height, decoded.taken_at, decoded.gps,
                         decoded.content_hash, w, h, self._phash(decoded.rgb) if self._phash else None)
        item_ids = self._store.create_items(job, meta)
        for item_id in item_ids:
            self._cache.write_working(item_id, decoded.rgb)
        self._cache.write_thumbnail(job.media_id, decoded.rgb)
        self._store.finish_derive(job, item_ids, self._next_stages())

    # -- faces: one model loaded, decode pool feeds one inference loop -------------------------------

    def run_faces(self) -> StageReport:
        report = StageReport(Stage.FACES)
        analyzer: FaceAnalyzer | None = None
        try:
            with ThreadPoolExecutor(self._s.decode_workers, thread_name_prefix="load") as pool:
                while not self._should_stop():
                    jobs = self._store.lease_jobs(Stage.FACES, self._s.batch_size,
                                                  self._s.lease_seconds, self._s.max_attempts)
                    if not jobs:
                        break
                    if analyzer is None:
                        analyzer = self._make_analyzer()  # loaded only when there is work
                        self._suppressed = self._suppressed_source(analyzer.model_id)
                    images = [pool.submit(self._working_image, j) for j in jobs]
                    for job, image in zip(jobs, images):
                        ok = self._guarded(job, lambda j: self._faces_one(j, image.result(), analyzer), report)
                        self._count(report, ok)
                    self._cleanup([j.item_id for j in jobs if j.item_id is not None])
        finally:
            if analyzer is not None:
                analyzer.close()
        return report

    # -- embed: image embeddings for search, same pattern as faces --------------------------------------

    def run_embed(self) -> StageReport:
        report = StageReport(Stage.EMBED)
        if self._make_embedder is None:
            return report
        embedder: Embedder | None = None
        try:
            with ThreadPoolExecutor(self._s.decode_workers, thread_name_prefix="load") as pool:
                while not self._should_stop():
                    jobs = self._store.lease_jobs(Stage.EMBED, self._s.batch_size,
                                                  self._s.lease_seconds, self._s.max_attempts)
                    if not jobs:
                        break
                    if embedder is None:
                        embedder = self._make_embedder()
                    loads = [pool.submit(self._working_image, j) for j in jobs]
                    ready, images = [], []
                    for job, load in zip(jobs, loads):  # a broken image fails alone, not the whole batch
                        if self._guarded(job, lambda j, f=load: images.append(f.result()), report):
                            ready.append(job)
                        else:
                            self._count(report, False)
                    if ready:
                        self._embed_batch(embedder, ready, images, report)
                    self._cleanup([j.item_id for j in jobs if j.item_id is not None])
        finally:
            if embedder is not None:
                embedder.close()
        return report

    def _embed_batch(self, embedder: Embedder, jobs: list[Job], images: list[np.ndarray],
                     report: StageReport) -> None:
        try:
            vectors = embedder.embed_images(images)
        except Exception as exc:
            log.exception("embedding batch failed")
            for job in jobs:
                self._fail(job, f"{type(exc).__name__}: {exc}", permanent=False, report=report)
                self._count(report, False)
            return
        for job, vec in zip(jobs, vectors):
            ok = self._guarded(job, lambda j, v=vec: self._store.finish_embed(j, v), report)
            self._count(report, ok)

    # -- ocr: text in pictures likely to have it (runs after organising, which decides what is a screenshot) ---

    def run_ocr(self) -> StageReport:
        report = StageReport(Stage.OCR)
        if not self._ocr_model_id:
            return report
        self._store.enqueue_ocr(self._ocr_model_id, self._ocr_all)
        reader: TextReader | None = None
        try:
            with ThreadPoolExecutor(self._s.decode_workers, thread_name_prefix="load") as pool:
                while not self._should_stop():
                    jobs = self._store.lease_jobs(Stage.OCR, self._s.batch_size,
                                                  self._s.lease_seconds, self._s.max_attempts)
                    if not jobs:
                        break
                    if reader is None:
                        reader = self._make_reader()
                    images = [pool.submit(self._working_image, j) for j in jobs]
                    for job, image in zip(jobs, images):
                        ok = self._guarded(job, lambda j, im=image: self._store.finish_ocr(j, reader.read(im.result())), report)
                        self._count(report, ok)
                    self._cleanup([j.item_id for j in jobs if j.item_id is not None])
        finally:
            if reader is not None:
                reader.close()
        return report

    def _working_image(self, job: Job) -> np.ndarray:
        try:
            return self._cache.read_working(job.item_id)
        except FileNotFoundError:
            # Working images are removed once an item is fully processed; rebuild from the original.
            return self._decoder.decode(self._store.media_path(job.media_id)).rgb

    def _faces_one(self, job: Job, image: np.ndarray, analyzer: FaceAnalyzer) -> None:
        faces = [f for f in analyzer.analyze(image) if self._gate.accepts(f) and not self._is_suppressed(f)]
        qualities = [self._gate.score(f) for f in faces]
        face_ids = self._store.finish_faces(job, faces, qualities, analyzer.detector_id)
        for face_id, face in zip(face_ids, faces):
            self._cache.write_face_crop(face_id, image, face.bbox)

    def _is_suppressed(self, face) -> bool:
        """Faces of people the user asked to forget are not stored again on a rescan."""
        if face.embedding is None or len(self._suppressed) == 0:
            return False
        return float((self._suppressed @ face.embedding).max()) >= self._suppress_similarity

    def _cleanup(self, item_ids: list[int]) -> None:
        if self._s.keep_working_images:
            return
        open_items = self._store.items_with_open_jobs(item_ids)
        for item_id in item_ids:
            if item_id not in open_items:
                self._cache.delete_working(item_id)

    # -- shared ----------------------------------------------------------------------------------------

    def _guarded(self, job: Job, fn: Callable[[Job], None], report: StageReport) -> bool:
        try:
            fn(job)
            return True
        except PermanentMediaError as exc:
            self._fail(job, str(exc), permanent=True, report=report)
        except Exception as exc:  # transient by default: retried with backoff, then dead
            log.exception("job %s (%s) failed", job.id, job.stage)
            self._fail(job, f"{type(exc).__name__}: {exc}", permanent=False, report=report)
        return False

    def _fail(self, job: Job, error: str, permanent: bool, report: StageReport) -> None:
        self._store.fail_job(job, error, permanent, self._s.max_attempts, self._s.backoff_seconds)
        if len(report.errors) < 50:
            report.errors.append(f"media {job.media_id}: {error}")

    def _count(self, report: StageReport, ok: bool) -> None:
        if ok:
            report.done += 1
        else:
            report.failed += 1
        self._progress(report.stage, report.done, report.failed)
