"""SQLite + in-memory vectors for search.

Vectors are canonical in item_embedding. For <50k photos, an exact dot product over an in-memory matrix
(~150 MB at 768 dims) answers a query in milliseconds, so no approximate index (sqlite-vec) is needed yet.
The matrix reloads only when the table changes.
"""

from __future__ import annotations

import json
import re
import threading

import numpy as np

from facegroup.adapters.storage.sqlite_store import SqliteBase
from facegroup.ports.search import SearchFilters

_WORD = re.compile(r"\w+", re.UNICODE)


class SqliteSearchStore(SqliteBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: dict[str, tuple[tuple[int, int], np.ndarray, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    def candidate_items(self, f: SearchFilters) -> tuple[np.ndarray, np.ndarray]:
        sql = ["SELECT i.id, i.media_id FROM item i JOIN media m ON m.id=i.media_id WHERE m.file_state='active'"]
        args: list = []
        for pid in f.person_ids:
            sql.append("AND EXISTS (SELECT 1 FROM face f JOIN item i2 ON i2.id=f.item_id "
                       "WHERE i2.media_id=m.id AND f.person_id=?)")
            args.append(pid)
        date = "COALESCE(m.taken_at, strftime('%Y-%m-%dT%H:%M:%S', m.mtime, 'unixepoch'))"
        if f.date_from:
            sql.append(f"AND {date} >= ?")
            args.append(f.date_from)
        if f.date_to:
            sql.append(f"AND {date} < date(?, '+1 day')")
            args.append(f.date_to)
        if f.root_id is not None:
            sql.append("AND m.root_id=?")
            args.append(f.root_id)
        if f.folder:
            prefix = f.folder.strip("/").replace("\\", "/") + "/"
            sql.append("AND substr(m.rel_path, 1, ?)=?")
            args += [len(prefix), prefix]
        rows = self._read().execute(" ".join(sql), args).fetchall()
        return np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64)

    def _matrix(self, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        stamp = tuple(self._read().execute(
            "SELECT COUNT(*), COALESCE(SUM(item_id), 0) FROM item_embedding WHERE model_id=?", (model_id,)
        ).fetchone())
        with self._cache_lock:
            cached = self._cache.get(model_id)
            if cached and cached[0] == stamp:
                return cached[1], cached[2]
            rows = self._read().execute(
                "SELECT item_id, vector FROM item_embedding WHERE model_id=? ORDER BY item_id", (model_id,)
            ).fetchall()
            ids = np.array([r[0] for r in rows], np.int64)
            mat = (np.stack([np.frombuffer(r[1], np.float32) for r in rows]) if rows
                   else np.zeros((0, 1), np.float32))
            self._cache[model_id] = (stamp, ids, mat)
            return ids, mat

    def item_vectors(self, model_id: str, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids, mat = self._matrix(model_id)
        if len(ids) == 0 or len(item_ids) == 0:
            return np.zeros(0, np.int64), mat[:0]
        pos = np.searchsorted(ids, item_ids)
        pos = np.clip(pos, 0, len(ids) - 1)
        found = ids[pos] == item_ids
        return item_ids[found], mat[pos[found]]

    def keyword_media(self, query: str, media_ids: np.ndarray | None, limit: int) -> list[int]:
        words = _WORD.findall(query.lower())
        if not words:
            return []
        # Every word must match (as a prefix: "goa" finds "goa2023"); quoted so user text is never FTS syntax.
        match = " AND ".join(f'"{w}"*' for w in words)
        rows = self._read().execute(
            "SELECT media_id FROM media_text WHERE media_text MATCH ? ORDER BY rank LIMIT ?",
            (match, limit * 4 if media_ids is not None else limit),
        ).fetchall()
        hits = [int(r[0]) for r in rows]
        if media_ids is not None:
            allowed = set(media_ids.tolist())
            hits = [m for m in hits if m in allowed]
        return hits[:limit]

    def media_vector(self, media_id: int, model_id: str) -> np.ndarray | None:
        row = self._read().execute(
            "SELECT e.vector FROM item i JOIN item_embedding e ON e.item_id=i.id "
            "WHERE i.media_id=? AND e.model_id=? ORDER BY i.t_ms LIMIT 1", (media_id, model_id)
        ).fetchone()
        return np.frombuffer(row[0], np.float32) if row else None

    def newest_first(self, media_ids: np.ndarray) -> list[int]:
        if len(media_ids) == 0:
            return []
        # One JSON parameter instead of a temp table: read connections are query-only.
        rows = self._read().execute(
            "SELECT m.id FROM media m WHERE m.id IN (SELECT value FROM json_each(?)) "
            "ORDER BY COALESCE(m.taken_at, strftime('%Y-%m-%dT%H:%M:%S', m.mtime, 'unixepoch')) DESC, m.id DESC",
            (json.dumps([int(m) for m in media_ids]),),
        ).fetchall()
        return [r[0] for r in rows]

    def photo_info(self, media_id: int) -> dict | None:
        r = self._read().execute(
            "SELECT m.id, r.path AS root, m.rel_path, m.taken_at, m.width, m.height, m.size, m.gps_lat, m.gps_lon "
            "FROM media m JOIN root r ON r.id=m.root_id WHERE m.id=?", (media_id,)).fetchone()
        if r is None:
            return None
        faces = self._read().execute(
            "SELECT f.id, f.person_id, p.name, p.kind, f.assigned_by FROM face f JOIN item i ON i.id=f.item_id "
            "LEFT JOIN person p ON p.id=f.person_id WHERE i.media_id=? ORDER BY f.x1", (media_id,)).fetchall()
        return {**dict(r), "faces": [dict(f) for f in faces]}

    def top_folders(self, limit: int = 200) -> list[tuple[int, str]]:
        rows = self._read().execute(
            "SELECT DISTINCT root_id, substr(rel_path, 1, instr(rel_path, '/') - 1) AS folder FROM media "
            "WHERE instr(rel_path, '/') > 0 ORDER BY folder LIMIT ?", (limit,)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def embedding_coverage(self, model_id: str) -> tuple[int, int]:
        c = self._read()
        done = c.execute("SELECT COUNT(*) FROM item_embedding WHERE model_id=?", (model_id,)).fetchone()[0]
        total = c.execute("SELECT COUNT(*) FROM item").fetchone()[0]
        return done, total
