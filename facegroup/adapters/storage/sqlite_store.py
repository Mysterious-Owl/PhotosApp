"""SQLite implementation of the Store port. Reads use per-thread read-only connections; writes go to DbWriter."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np

from facegroup.adapters.storage.base import JSON_IDS, SqliteBase, iso, json_ids
from facegroup.domain.entities import DetectedFace, FileState, Job, JobState, MediaRecord, Stage
from facegroup.domain.policies import match_boxes
from facegroup.ports.store import ImageMeta, Root, ScanPlan


def _root(r: sqlite3.Row) -> Root:
    return Root(r["id"], Path(r["path"]), bool(r["fetch_online_only"]))


def _mark_done(c: sqlite3.Connection, job_id: int) -> None:
    c.execute("UPDATE job SET state='done', lease_until=NULL, error=NULL WHERE id=?", (job_id,))


class SqliteStore(SqliteBase):

    # -- roots and scanning -------------------------------------------------------------------

    def add_root(self, path: Path, fetch_online_only: bool = False) -> Root:
        key = str(path)

        def op(c: sqlite3.Connection) -> Root:
            c.execute(
                "INSERT INTO root(path, fetch_online_only) VALUES(?, ?) "
                "ON CONFLICT(path) DO UPDATE SET fetch_online_only=excluded.fetch_online_only, enabled=1",
                (key, int(fetch_online_only)),
            )
            return _root(c.execute("SELECT id, path, fetch_online_only FROM root WHERE path=?", (key,)).fetchone())

        return self._write(op)

    def roots(self) -> list[Root]:
        rows = self._read().execute(
            "SELECT id, path, fetch_online_only FROM root WHERE enabled=1 ORDER BY id"
        ).fetchall()
        return [_root(r) for r in rows]

    def known_media(self, root_id: int) -> dict[str, MediaRecord]:
        rows = self._read().execute(
            "SELECT id, root_id, rel_path, size, mtime, file_state, content_hash, missing_since "
            "FROM media WHERE root_id=?",
            (root_id,),
        ).fetchall()
        return {
            r["rel_path"]: MediaRecord(
                r["id"], r["root_id"], r["rel_path"], r["size"], r["mtime"],
                FileState(r["file_state"]), r["content_hash"], r["missing_since"],
            )
            for r in rows
        }

    def apply_scan(self, plan: ScanPlan) -> int:
        now = iso(self._now())

        def requeue_derive(c: sqlite3.Connection, media_id: int) -> None:
            # Old items (and their faces and jobs) describe the old file contents.
            c.execute("DELETE FROM item WHERE media_id=?", (media_id,))
            c.execute("DELETE FROM job WHERE media_id=?", (media_id,))
            c.execute("INSERT INTO job(media_id, stage) VALUES(?, ?)", (media_id, Stage.DERIVE.value))

        def op(c: sqlite3.Connection) -> int:
            for items, state, queue in ((plan.new, FileState.ACTIVE, True),
                                        (plan.new_online_only, FileState.ONLINE_ONLY, False)):
                for f in items:
                    media_id = c.execute(
                        "INSERT INTO media(root_id, rel_path, size, mtime, file_state) VALUES(?,?,?,?,?)",
                        (plan.root_id, f.rel_path, f.size, f.mtime, state.value),
                    ).lastrowid
                    if queue:
                        c.execute("INSERT INTO job(media_id, stage) VALUES(?, ?)", (media_id, Stage.DERIVE.value))
            for media_id, f in plan.changed + plan.hydrated:
                c.execute(
                    "UPDATE media SET size=?, mtime=?, content_hash=NULL, file_state=?, missing_since=NULL "
                    "WHERE id=?",
                    (f.size, f.mtime, FileState.CHANGED.value, media_id),
                )
                requeue_derive(c, media_id)
            if plan.reappeared:
                c.execute(
                    f"UPDATE media SET file_state=?, missing_since=NULL WHERE id IN {JSON_IDS}",
                    (FileState.ACTIVE.value, json_ids(plan.reappeared)),
                )
            if plan.became_online_only:
                c.execute(
                    f"UPDATE media SET file_state=? WHERE id IN {JSON_IDS}",
                    (FileState.ONLINE_ONLY.value, json_ids(plan.became_online_only)),
                )
            for media_id, f in plan.moved:
                c.execute(
                    "UPDATE media SET rel_path=?, size=?, mtime=?, file_state=?, missing_since=NULL WHERE id=?",
                    (f.rel_path, f.size, f.mtime, FileState.ACTIVE.value, media_id),
                )
            if plan.missing:
                c.execute(
                    f"UPDATE media SET file_state=?, missing_since=COALESCE(missing_since, ?) WHERE id IN {JSON_IDS}",
                    (FileState.MISSING.value, now, json_ids(plan.missing)),
                )
            removed = 0
            if plan.remove_missing_before:
                removed = c.execute(
                    "DELETE FROM media WHERE root_id=? AND file_state=? AND missing_since < ?",
                    (plan.root_id, FileState.MISSING.value, plan.remove_missing_before),
                ).rowcount
            return removed

        return self._write(op)

    def media_path(self, media_id: int) -> Path:
        row = self._read().execute(
            "SELECT r.path AS root, m.rel_path FROM media m JOIN root r ON r.id=m.root_id WHERE m.id=?",
            (media_id,),
        ).fetchone()
        if row is None:
            raise KeyError(media_id)
        return Path(row["root"]) / Path(row["rel_path"])

    # -- jobs -----------------------------------------------------------------------------------

    def release_stale_leases(self, max_attempts: int) -> int:
        def op(c: sqlite3.Connection) -> int:
            c.execute(
                "UPDATE job SET state='dead', lease_until=NULL, "
                "error=COALESCE(error || '; ', '') || 'process stopped while running this job' "
                "WHERE state='running' AND attempts>=?",
                (max_attempts,),
            )
            return c.execute(
                "UPDATE job SET state='pending', lease_until=NULL WHERE state='running'"
            ).rowcount

        return self._write(op)

    def lease_jobs(self, stage: Stage, limit: int, lease_seconds: int, max_attempts: int) -> list[Job]:
        now_t = self._now()
        now, until = iso(now_t), iso(now_t + timedelta(seconds=lease_seconds))

        def op(c: sqlite3.Connection) -> list[Job]:
            # Expired leases count as attempts, so a file that kills the worker cannot loop forever.
            c.execute(
                "UPDATE job SET state='dead', lease_until=NULL, "
                "error=COALESCE(error || '; ', '') || 'lease expired' "
                "WHERE stage=? AND state='running' AND lease_until<? AND attempts>=?",
                (stage.value, now, max_attempts),
            )
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM job WHERE stage=? AND ("
                    " state='pending'"
                    " OR (state='retry_wait' AND not_before<=?)"
                    " OR (state='running' AND lease_until<?)"
                    ") ORDER BY id LIMIT ?",
                    (stage.value, now, now, limit),
                )
            ]
            if not ids:
                return []
            rows = c.execute(
                f"UPDATE job SET state='running', attempts=attempts+1, lease_until=?, not_before=NULL "
                f"WHERE id IN {JSON_IDS} RETURNING id, media_id, item_id, stage, model_id, attempts",
                (until, json_ids(ids)),
            ).fetchall()
            return sorted(
                (Job(r["id"], r["media_id"], r["item_id"], Stage(r["stage"]), r["model_id"], r["attempts"])
                 for r in rows),
                key=lambda j: j.id,
            )

        return self._write(op)

    def create_items(self, job: Job, meta: ImageMeta) -> list[int]:
        def op(c: sqlite3.Connection) -> list[int]:
            lat, lon = meta.gps if meta.gps else (None, None)
            c.execute(
                "UPDATE media SET width=?, height=?, taken_at=?, gps_lat=?, gps_lon=?, content_hash=?, phash=? WHERE id=?",
                (meta.width, meta.height, meta.taken_at, lat, lon, meta.content_hash, meta.phash, job.media_id),
            )
            # Idempotent: a re-run after a crash replaces the items it made last time.
            c.execute("DELETE FROM item WHERE media_id=?", (job.media_id,))
            cur = c.execute(
                "INSERT INTO item(media_id, t_ms, width, height) VALUES(?, 0, ?, ?)",
                (job.media_id, meta.working_width, meta.working_height),
            )
            return [cur.lastrowid]

        return self._write(op)

    def finish_derive(self, job: Job, item_ids: list[int], next_stages: list[tuple[Stage, str]]) -> None:
        def op(c: sqlite3.Connection) -> None:
            _mark_done(c, job.id)
            c.execute(
                "UPDATE media SET file_state=? WHERE id=? AND file_state=?",
                (FileState.ACTIVE.value, job.media_id, FileState.CHANGED.value),
            )
            c.executemany(
                "INSERT OR IGNORE INTO job(media_id, item_id, stage, model_id) VALUES(?,?,?,?)",
                [(job.media_id, i, stage.value, model_id) for i in item_ids for stage, model_id in next_stages],
            )

        self._write(op)

    def finish_faces(self, job: Job, faces: list[DetectedFace], qualities: list[float],
                     detector_id: str) -> list[int]:
        assert job.item_id is not None

        def op(c: sqlite3.Connection) -> list[int]:
            old = c.execute(
                "SELECT id, x1, y1, x2, y2, person_id FROM face WHERE item_id=? AND detector_id=?",
                (job.item_id, detector_id),
            ).fetchall()
            matches = match_boxes([(r["x1"], r["y1"], r["x2"], r["y2"]) for r in old], [f.bbox for f in faces])
            kept: set[int] = set()
            ids: list[int] = []
            for i, (face, quality) in enumerate(zip(faces, qualities)):
                values = (*face.bbox, face.landmarks.astype(np.float32).tobytes(), face.det_score, quality)
                if i in matches:  # same face re-detected: keep the row, so labels survive
                    face_id = old[matches[i]]["id"]
                    c.execute(
                        "UPDATE face SET x1=?, y1=?, x2=?, y2=?, landmarks=?, det_score=?, quality=? WHERE id=?",
                        (*values, face_id),
                    )
                else:
                    face_id = c.execute(
                        "INSERT INTO face(item_id, x1, y1, x2, y2, landmarks, det_score, quality, detector_id) "
                        "VALUES(?,?,?,?,?,?,?,?,?)",
                        (job.item_id, *values, detector_id),
                    ).lastrowid
                kept.add(face_id)
                ids.append(face_id)
                if face.embedding is not None:
                    c.execute(
                        "INSERT OR REPLACE INTO face_embedding(face_id, model_id, vector) VALUES(?,?,?)",
                        (face_id, job.model_id, face.embedding.astype(np.float32).tobytes()),
                    )
            # Old detections that vanished: drop them unless a person was assigned (never lose a label).
            for r in old:
                if r["id"] not in kept and r["person_id"] is None:
                    c.execute("DELETE FROM face WHERE id=?", (r["id"],))
            _mark_done(c, job.id)
            return ids

        return self._write(op)

    def finish_embed(self, job: Job, vector: np.ndarray) -> None:
        def op(c: sqlite3.Connection) -> None:
            c.execute("INSERT OR REPLACE INTO item_embedding(item_id, model_id, vector) VALUES(?, ?, ?)",
                      (job.item_id, job.model_id, np.asarray(vector, np.float32).tobytes()))
            _mark_done(c, job.id)

        self._write(op)

    def finish_ocr(self, job: Job, text: str) -> None:
        def op(c: sqlite3.Connection) -> None:
            c.execute("DELETE FROM item_text WHERE item_id=? AND source='ocr'", (job.item_id,))
            if text:
                c.execute("INSERT INTO item_text(item_id, source, model_id, body) VALUES(?, 'ocr', ?, ?)",
                          (job.item_id, job.model_id, text))
            _mark_done(c, job.id)

        self._write(op)

    def enqueue_ocr(self, model_id: str, all_photos: bool) -> int:
        """Queue text reading for pictures likely to contain text (screenshots, documents, receipts, memes),
        or for everything when `all_photos` is on. Each item once per model."""
        where = "" if all_photos else "AND COALESCE(mc.category, 'photo') IN ('screenshot', 'document', 'receipt', 'meme')"

        def op(c: sqlite3.Connection) -> int:
            return c.execute(
                "INSERT OR IGNORE INTO job(media_id, item_id, stage, model_id) "
                "SELECT i.media_id, i.id, 'ocr', ? FROM item i JOIN media m ON m.id=i.media_id "
                "LEFT JOIN media_category mc ON mc.media_id=m.id "
                f"WHERE m.file_state='active' {where} AND NOT EXISTS ("
                "  SELECT 1 FROM job j WHERE j.item_id=i.id AND j.stage='ocr' AND j.model_id=?)",
                (model_id, model_id),
            ).rowcount

        return self._write(op)

    def enqueue_missing(self, stage: Stage, model_id: str) -> int:
        """Queue `stage` for every item of an active photo that has no job for this model yet."""
        def op(c: sqlite3.Connection) -> int:
            return c.execute(
                "INSERT OR IGNORE INTO job(media_id, item_id, stage, model_id) "
                "SELECT i.media_id, i.id, ?, ? FROM item i JOIN media m ON m.id=i.media_id "
                "WHERE m.file_state='active' AND NOT EXISTS ("
                "  SELECT 1 FROM job j WHERE j.item_id=i.id AND j.stage=? AND j.model_id=?)",
                (stage.value, model_id, stage.value, model_id),
            ).rowcount

        return self._write(op)

    def fail_job(self, job: Job, error: str, permanent: bool, max_attempts: int,
                 backoff_seconds: int) -> None:
        dead = permanent or job.attempts >= max_attempts
        not_before = None if dead else iso(self._now() + timedelta(seconds=backoff_seconds * 2 ** (job.attempts - 1)))
        state = JobState.DEAD if dead else JobState.RETRY_WAIT

        def op(c: sqlite3.Connection) -> None:
            c.execute(
                "UPDATE job SET state=?, not_before=?, lease_until=NULL, error=? WHERE id=?",
                (state.value, not_before, error[:2000], job.id),
            )

        self._write(op)

    def items_with_open_jobs(self, item_ids: list[int]) -> set[int]:
        if not item_ids:
            return set()
        rows = self._read().execute(
            f"SELECT DISTINCT item_id FROM job WHERE item_id IN {JSON_IDS} AND state NOT IN ('done', 'dead')",
            (json_ids(item_ids),),
        ).fetchall()
        return {r[0] for r in rows}

    def retry_dead(self) -> int:
        return self._write(
            lambda c: c.execute(
                "UPDATE job SET state='pending', attempts=0, error=NULL, not_before=NULL WHERE state='dead'"
            ).rowcount
        )

    def dead_jobs(self, limit: int = 20) -> list[tuple[str, str, str]]:
        rows = self._read().execute(
            "SELECT r.path AS root, m.rel_path, j.stage, j.error FROM job j "
            "JOIN media m ON m.id=j.media_id JOIN root r ON r.id=m.root_id "
            "WHERE j.state='dead' ORDER BY j.id LIMIT ?",
            (limit,),
        ).fetchall()
        return [(str(Path(r["root"]) / r["rel_path"]), r["stage"], r["error"] or "") for r in rows]

    def status(self) -> dict[str, dict[str, int]]:
        c = self._read()
        media = {r[0]: r[1] for r in c.execute("SELECT file_state, COUNT(*) FROM media GROUP BY file_state")}
        jobs: dict[str, int] = {}
        for stage, state, n in c.execute("SELECT stage, state, COUNT(*) FROM job GROUP BY stage, state"):
            jobs[f"{stage}.{state}"] = n
        faces = {
            "faces": c.execute("SELECT COUNT(*) FROM face").fetchone()[0],
            "items": c.execute("SELECT COUNT(*) FROM item").fetchone()[0],
        }
        return {"media": media, "jobs": jobs, "counts": faces}
