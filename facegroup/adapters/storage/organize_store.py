"""SQLite storage for organising: categories, duplicates, stacks and per-photo flags (all inside FaceGroup)."""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from facegroup.adapters.storage.sqlite_store import SqliteBase


class SqliteOrganizeStore(SqliteBase):
    def organize_rows(self, model_id: str) -> tuple[list[dict], np.ndarray]:
        """Active photos with what organising needs, and the image embedding of each (zeros if not embedded)."""
        rows = self._read().execute(
            "SELECT m.id, m.root_id, m.rel_path, m.width, m.height, m.taken_at, m.mtime, m.size, m.content_hash, "
            "m.phash, c.category, c.source, "
            "(SELECT e.vector FROM item i JOIN item_embedding e ON e.item_id=i.id AND e.model_id=? "
            " WHERE i.media_id=m.id ORDER BY i.t_ms LIMIT 1) AS vec, "
            "(SELECT MAX(f.quality) FROM item i JOIN face f ON f.item_id=i.id WHERE i.media_id=m.id) AS face_q "
            "FROM media m LEFT JOIN media_category c ON c.media_id=m.id "
            "WHERE m.file_state='active' AND m.width IS NOT NULL ORDER BY m.id",
            (model_id,),
        ).fetchall()
        out, vecs, dim = [], [], 0
        for r in rows:
            d = dict(r)
            v = d.pop("vec")
            if v is not None:
                arr = np.frombuffer(v, np.float32)
                dim = len(arr)
            else:
                arr = None
            vecs.append(arr)
            out.append(d)
        matrix = np.zeros((len(out), dim or 1), np.float32)
        for i, v in enumerate(vecs):
            if v is not None:
                matrix[i] = v
                out[i]["has_vec"] = True
            else:
                out[i]["has_vec"] = False
        return out, matrix

    def set_phashes(self, values: list[tuple[int, int]]) -> None:
        if values:
            self._write(lambda c: c.executemany("UPDATE media SET phash=? WHERE id=?", [(h, m) for m, h in values]))

    # -- categories ---------------------------------------------------------------------------------------

    def write_categories(self, rows: list[tuple[int, str, float, str]]) -> None:
        """(media_id, category, score, model_id). Never overwrites a choice the user made."""
        def op(c: sqlite3.Connection) -> None:
            c.executemany(
                "INSERT INTO media_category(media_id, category, score, source, model_id) VALUES(?, ?, ?, 'auto', ?) "
                "ON CONFLICT(media_id) DO UPDATE SET category=excluded.category, score=excluded.score, "
                "model_id=excluded.model_id WHERE media_category.source='auto'",
                rows,
            )
        if rows:
            self._write(op)

    def set_user_category(self, media_id: int, category: str) -> None:
        self._write(lambda c: c.execute(
            "INSERT INTO media_category(media_id, category, source) VALUES(?, ?, 'user') "
            "ON CONFLICT(media_id) DO UPDATE SET category=excluded.category, source='user', score=NULL",
            (media_id, category)))

    def category_counts(self) -> dict[str, int]:
        rows = self._read().execute(
            "SELECT COALESCE(c.category, 'photo'), COUNT(*) FROM media m LEFT JOIN media_category c ON c.media_id=m.id "
            "WHERE m.file_state='active' AND m.hidden=0 GROUP BY 1").fetchall()
        return {r[0]: r[1] for r in rows}

    # -- duplicates and stacks ------------------------------------------------------------------------------

    def replace_duplicates(self, sets: list[tuple[str, list[int], int]]) -> None:
        """(kind, member media ids, keeper id)."""
        def op(c: sqlite3.Connection) -> None:
            c.execute("DELETE FROM duplicate_member")
            for set_id, (kind, members, keeper) in enumerate(sets, start=1):
                c.executemany("INSERT INTO duplicate_member(media_id, set_id, kind, keeper) VALUES(?, ?, ?, ?)",
                              [(m, set_id, kind, int(m == keeper)) for m in members])
        self._write(op)

    def duplicate_sets(self) -> list[dict]:
        rows = self._read().execute(
            "SELECT d.set_id, d.kind, d.keeper, m.id, r.path AS root, m.rel_path, m.width, m.height, m.size, m.hidden "
            "FROM duplicate_member d JOIN media m ON m.id=d.media_id JOIN root r ON r.id=m.root_id "
            "ORDER BY d.set_id, d.keeper DESC, m.id").fetchall()
        sets: dict[int, dict] = {}
        for r in rows:
            s = sets.setdefault(r["set_id"], {"id": r["set_id"], "kind": r["kind"], "members": []})
            s["members"].append(dict(r))
        return list(sets.values())

    def duplicate_set_of(self, media_id: int) -> list[int]:
        rows = self._read().execute(
            "SELECT media_id FROM duplicate_member WHERE set_id=(SELECT set_id FROM duplicate_member WHERE media_id=?)",
            (media_id,)).fetchall()
        return [r[0] for r in rows]

    def stack_picks(self) -> set[int]:
        return {r[0] for r in self._read().execute("SELECT media_id FROM stack_pick")}

    def replace_stacks(self, stacks: list[tuple[list[int], int]]) -> None:
        """(member media ids, top media id)."""
        def op(c: sqlite3.Connection) -> None:
            c.execute("DELETE FROM stack_member")
            for stack_id, (members, top) in enumerate(stacks, start=1):
                c.executemany("INSERT INTO stack_member(media_id, stack_id, top) VALUES(?, ?, ?)",
                              [(m, stack_id, int(m == top)) for m in members])
        self._write(op)

    def set_stack_top(self, media_id: int) -> None:
        def op(c: sqlite3.Connection) -> None:
            members = [r[0] for r in c.execute(
                "SELECT media_id FROM stack_member WHERE stack_id=(SELECT stack_id FROM stack_member WHERE media_id=?)",
                (media_id,))]
            if not members:
                return
            c.execute(f"DELETE FROM stack_pick WHERE media_id IN ({','.join('?' * len(members))})", members)
            c.execute("INSERT INTO stack_pick(media_id) VALUES(?)", (media_id,))
            c.execute(f"UPDATE stack_member SET top=(media_id=?) WHERE media_id IN ({','.join('?' * len(members))})",
                      (media_id, *members))
        self._write(op)

    def stack_of(self, media_id: int) -> list[tuple[int, bool]]:
        rows = self._read().execute(
            "SELECT media_id, top FROM stack_member WHERE stack_id=(SELECT stack_id FROM stack_member WHERE media_id=?) "
            "ORDER BY top DESC, media_id", (media_id,)).fetchall()
        return [(r[0], bool(r[1])) for r in rows]

    # -- per-photo flags ------------------------------------------------------------------------------------

    def set_flags(self, media_ids: list[int], favorite: bool | None = None, rating: int | None | str = "keep",
                  hidden: bool | None = None) -> None:
        sets, args = [], []
        if favorite is not None:
            sets.append("favorite=?")
            args.append(int(favorite))
        if rating != "keep":
            sets.append("rating=?")
            args.append(rating)
        if hidden is not None:
            sets.append("hidden=?")
            args.append(int(hidden))
        if not sets or not media_ids:
            return
        self._write(lambda c: c.execute(
            f"UPDATE media SET {', '.join(sets)} WHERE id IN (SELECT value FROM json_each(?))",
            (*args, json.dumps([int(m) for m in media_ids]))))

    def flags(self, media_id: int) -> dict:
        r = self._read().execute(
            "SELECT m.favorite, m.rating, m.hidden, COALESCE(c.category, 'photo') AS category, c.source "
            "FROM media m LEFT JOIN media_category c ON c.media_id=m.id WHERE m.id=?", (media_id,)).fetchone()
        return dict(r) if r else {}

    def hidden_count(self) -> int:
        return self._read().execute("SELECT COUNT(*) FROM media WHERE hidden=1 AND file_state='active'").fetchone()[0]
