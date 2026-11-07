"""SQLite + in-memory vectors for search.

Vectors are canonical in item_embedding. For <50k photos, an exact dot product over an in-memory matrix
(~150 MB at 768 dims) answers a query in milliseconds, so no approximate index (sqlite-vec) is needed yet.
The matrix reloads only when the table changes (a trigger-kept counter, migration 0008).
"""

from __future__ import annotations

import re
import threading

import numpy as np

from facegroup.adapters.storage.base import JSON_IDS, SqliteBase, json_ids, stack_vectors
from facegroup.ports.search import SearchFilters

_WORD = re.compile(r"\w+", re.UNICODE)
# When a photo was taken: EXIF date if present, else the file's modification time.
DATE_SQL = "COALESCE(m.taken_at, strftime('%Y-%m-%dT%H:%M:%S', m.mtime, 'unixepoch'))"


class SqliteSearchStore(SqliteBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: dict[str, tuple[tuple[int, int], np.ndarray, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    def candidate_items(self, f: SearchFilters) -> tuple[np.ndarray, np.ndarray]:
        where, args = self._filter_sql(f)
        rows = self._read().execute(
            "SELECT i.id, i.media_id FROM item i JOIN media m ON m.id=i.media_id "
            "LEFT JOIN media_category mc ON mc.media_id=m.id WHERE " + where, args).fetchall()
        return np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64)

    def dated_media(self, f: SearchFilters) -> list[tuple[int, str]]:
        """(media_id, capture date or file date, ISO) for photos matching the filters, newest first. Used by the
        timeline, memories and events; the same filters as search, so the two always agree."""
        where, args = self._filter_sql(f)
        return [(r[0], r[1]) for r in self._read().execute(
            f"SELECT m.id, {DATE_SQL} AS d FROM media m LEFT JOIN media_category mc ON mc.media_id=m.id "
            f"WHERE {where} AND m.width IS NOT NULL ORDER BY d DESC, m.id DESC", args)]

    @staticmethod
    def _filter_sql(f: SearchFilters) -> tuple[str, list]:
        sql = ["m.file_state='active'"]
        args: list = []
        if f.kind == "hidden":
            sql.append("AND m.hidden=1")
        elif not f.show_hidden:
            sql.append("AND m.hidden=0")
        if f.kind in (None, "", "hidden"):
            pass
        elif f.kind == "photos":
            sql.append("AND COALESCE(mc.category, 'photo')='photo'")
        elif f.kind:
            sql.append("AND mc.category=?")
            args.append(f.kind)
        if f.favorites:
            sql.append("AND m.favorite=1")
        if f.min_rating:
            sql.append("AND m.rating>=?")
            args.append(f.min_rating)
        for pid in f.person_ids:
            sql.append("AND EXISTS (SELECT 1 FROM face f JOIN item i2 ON i2.id=f.item_id "
                       "WHERE i2.media_id=m.id AND f.person_id=?)")
            args.append(pid)
        if f.date_from:
            sql.append(f"AND {DATE_SQL} >= ?")
            args.append(f.date_from)
        if f.date_to:
            sql.append(f"AND {DATE_SQL} < date(?, '+1 day')")
            args.append(f.date_to)
        if f.root_id is not None:
            sql.append("AND m.root_id=?")
            args.append(f.root_id)
        if f.folder:
            prefix = f.folder.strip("/").replace("\\", "/") + "/"
            sql.append("AND substr(m.rel_path, 1, ?)=?")
            args += [len(prefix), prefix]
        return " ".join(sql), args

    def _matrix(self, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        stamp = self._read().execute("SELECT n FROM embedding_version").fetchone()[0]
        with self._cache_lock:
            cached = self._cache.get(model_id)
            if cached and cached[0] == stamp:
                return cached[1], cached[2]
            rows = self._read().execute(
                "SELECT item_id, vector FROM item_embedding WHERE model_id=? ORDER BY item_id", (model_id,)
            ).fetchall()
            ids = np.array([r[0] for r in rows], np.int64)
            mat = stack_vectors([r[1] for r in rows], 1)
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
        """Photos whose folder and file names match every word of the query, best first."""
        return self._fts_media("media_text", query, media_ids, limit)

    def text_media(self, query: str, media_ids: np.ndarray | None, limit: int) -> list[int]:
        """Photos whose text (OCR) matches every word of the query, best first."""
        return self._fts_media("ocr_fts", query, media_ids, limit)

    def _fts_media(self, table: str, query: str, media_ids: np.ndarray | None, limit: int) -> list[int]:
        words = _WORD.findall(query.lower())
        if not words:
            return []
        # Every word must match (as a prefix: "goa" finds "goa2023"); quoted so user text is never FTS syntax.
        match = " AND ".join(f'"{w}"*' for w in words)
        rows = self._read().execute(
            f"SELECT media_id FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
            (match, limit * 4 if media_ids is not None else limit),
        ).fetchall()
        hits = list(dict.fromkeys(int(r[0]) for r in rows))  # ocr_fts has a row per item, so a photo can repeat
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
        rows = self._read().execute(
            f"SELECT m.id FROM media m WHERE m.id IN {JSON_IDS} ORDER BY {DATE_SQL} DESC, m.id DESC",
            (json_ids(media_ids),),
        ).fetchall()
        return [r[0] for r in rows]

    def photo_info(self, media_id: int) -> dict | None:
        r = self._read().execute(
            "SELECT m.id, r.path AS root, m.rel_path, m.taken_at, m.width, m.height, m.size, m.gps_lat, m.gps_lon "
            "FROM media m JOIN root r ON r.id=m.root_id WHERE m.id=?", (media_id,)).fetchone()
        if r is None:
            return None
        faces = self._read().execute(
            "SELECT f.id, f.person_id, p.name, p.kind, f.assigned_by, f.x1, f.y1, f.x2, f.y2, "
            "i.width AS iw, i.height AS ih FROM face f JOIN item i ON i.id=f.item_id "
            "LEFT JOIN person p ON p.id=f.person_id WHERE i.media_id=? ORDER BY f.x1", (media_id,)).fetchall()
        return {**dict(r), "faces": [dict(f) for f in faces]}

    def top_folders(self, limit: int = 200) -> list[tuple[int, str]]:
        rows = self._read().execute(
            "SELECT DISTINCT root_id, substr(rel_path, 1, instr(rel_path, '/') - 1) AS folder FROM media "
            "WHERE instr(rel_path, '/') > 0 ORDER BY folder LIMIT ?", (limit,)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def stack_map(self, media_ids: list[int]) -> dict[int, tuple[int, bool, int]]:
        if not media_ids:
            return {}
        rows = self._read().execute(
            "SELECT s.media_id, s.stack_id, s.top, (SELECT COUNT(*) FROM stack_member x WHERE x.stack_id=s.stack_id) "
            f"FROM stack_member s WHERE s.media_id IN {JSON_IDS}", (json_ids(media_ids),)).fetchall()
        return {r[0]: (r[1], bool(r[2]), r[3]) for r in rows}

    # -- saved searches --------------------------------------------------------------------------------

    def saved_searches(self) -> list[dict]:
        return [dict(r) for r in self._read().execute("SELECT id, name, query FROM saved_search ORDER BY name")]

    def save_search(self, name: str, query: str) -> None:
        self._write(lambda c: c.execute(
            "INSERT INTO saved_search(name, query, created_at) VALUES(?, ?, datetime('now')) "
            "ON CONFLICT(name) DO UPDATE SET query=excluded.query", (name.strip(), query)))

    def delete_saved_search(self, search_id: int) -> None:
        self._write(lambda c: c.execute("DELETE FROM saved_search WHERE id=?", (search_id,)))

    def folder_of(self, media_ids: list[int]) -> dict[int, str]:
        rows = self._read().execute(
            f"SELECT id, rel_path FROM media WHERE id IN {JSON_IDS}", (json_ids(media_ids),)).fetchall()
        return {r[0]: r[1] for r in rows}

    def embedding_coverage(self, model_id: str) -> tuple[int, int]:
        c = self._read()
        done = c.execute("SELECT COUNT(*) FROM item_embedding WHERE model_id=?", (model_id,)).fetchone()[0]
        total = c.execute("SELECT COUNT(*) FROM item").fetchone()[0]
        return done, total
