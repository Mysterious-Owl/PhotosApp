"""SQLite implementation of the PeopleStore port."""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from facegroup.adapters.storage.sqlite_store import SqliteBase, _placeholders, iso
from facegroup.ports.people import FaceRow, PeopleChange, PersonSummary, SuggestionSummary


def _stack(blobs: list[bytes]) -> np.ndarray:
    if not blobs:
        return np.zeros((0, 512), np.float32)
    return np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])


def _chunks(ids: list[int], size: int = 500):
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


class SqlitePeopleStore(SqliteBase):
    # -- vectors for voting and clustering -----------------------------------------------------------

    def confirmed_faces(self, model_id: str):
        rows = self._read().execute(
            "SELECT f.id, f.person_id, e.vector FROM face f JOIN face_embedding e ON e.face_id=f.id "
            "WHERE e.model_id=? AND f.assigned_by='user' AND f.person_id IS NOT NULL ORDER BY f.id",
            (model_id,),
        ).fetchall()
        return (np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                _stack([r[2] for r in rows]))

    def votable_faces(self, model_id: str):
        rows = self._read().execute(
            "SELECT f.id, e.vector, f.person_id FROM face f JOIN face_embedding e ON e.face_id=f.id "
            "WHERE e.model_id=? AND (f.assigned_by IS NULL OR f.assigned_by='auto') ORDER BY f.id",
            (model_id,),
        ).fetchall()
        return np.array([r[0] for r in rows], np.int64), _stack([r[1] for r in rows]), [r[2] for r in rows]

    def unassigned_faces(self, model_id: str):
        rows = self._read().execute(
            "SELECT f.id, e.vector, f.quality FROM face f JOIN face_embedding e ON e.face_id=f.id "
            "WHERE e.model_id=? AND f.person_id IS NULL ORDER BY f.id",
            (model_id,),
        ).fetchall()
        return (np.array([r[0] for r in rows], np.int64), _stack([r[1] for r in rows]),
                np.array([r[2] for r in rows], np.float32))

    def rejections(self) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for face_id, person_id in self._read().execute("SELECT face_id, person_id FROM face_reject"):
            out.setdefault(face_id, set()).add(person_id)
        return out

    def suppressed_vectors(self, model_id: str) -> np.ndarray:
        rows = self._read().execute("SELECT vector FROM suppressed_face WHERE model_id=?", (model_id,)).fetchall()
        return _stack([r[0] for r in rows])

    # -- automatic assignment and suggestions ---------------------------------------------------------

    def write_auto_assignments(self, changes: list[tuple[int, int | None, float | None]]) -> int:
        if not changes:
            return 0

        def op(c: sqlite3.Connection) -> int:
            n = 0
            for face_id, person_id, score in changes:
                # The WHERE clause is the guarantee: a face the user decided is never touched.
                n += c.execute(
                    "UPDATE face SET person_id=?, assigned_by=?, assign_score=? "
                    "WHERE id=? AND (assigned_by IS NULL OR assigned_by='auto')",
                    (person_id, "auto" if person_id is not None else None, score, face_id),
                ).rowcount
            assigned = [f for f, p, _ in changes if p is not None]
            for chunk in _chunks(assigned):
                c.execute(f"DELETE FROM suggestion_face WHERE face_id IN ({_placeholders(len(chunk))})", chunk)
            return n

        return self._write(op)

    def replace_suggestions(self, groups: list[list[int]]) -> None:
        now = iso(self._now())

        def op(c: sqlite3.Connection) -> None:
            c.execute("DELETE FROM suggestion")
            for g in groups:
                sid = c.execute("INSERT INTO suggestion(created_at, size) VALUES(?, ?)", (now, len(g))).lastrowid
                c.executemany("INSERT INTO suggestion_face(suggestion_id, face_id) VALUES(?, ?)",
                              [(sid, f) for f in g])

        self._write(op)

    def suggestions(self, sample: int = 8) -> list[SuggestionSummary]:
        c = self._read()
        rows = c.execute(
            "SELECT s.id, COUNT(f.id) AS n FROM suggestion s "
            "JOIN suggestion_face sf ON sf.suggestion_id=s.id JOIN face f ON f.id=sf.face_id "
            "WHERE f.person_id IS NULL GROUP BY s.id HAVING n > 0 ORDER BY n DESC, s.id"
        ).fetchall()
        out = []
        for sid, n in rows:
            faces = [r[0] for r in c.execute(
                "SELECT f.id FROM suggestion_face sf JOIN face f ON f.id=sf.face_id "
                "WHERE sf.suggestion_id=? AND f.person_id IS NULL ORDER BY f.quality DESC LIMIT ?",
                (sid, sample),
            )]
            out.append(SuggestionSummary(sid, n, faces))
        return out

    def suggestion_faces(self, suggestion_id: int) -> list[int]:
        return [r[0] for r in self._read().execute(
            "SELECT f.id FROM suggestion_face sf JOIN face f ON f.id=sf.face_id "
            "WHERE sf.suggestion_id=? AND f.person_id IS NULL ORDER BY f.quality DESC",
            (suggestion_id,),
        )]

    # -- people ----------------------------------------------------------------------------------------

    _PERSON_SQL = (
        "SELECT p.id, p.name, p.kind, COUNT(f.id) AS faces, "
        "COALESCE(SUM(f.assigned_by='user'), 0) AS confirmed, COUNT(DISTINCT i.media_id) AS photos, "
        "COALESCE(p.cover_face_id, (SELECT f2.id FROM face f2 WHERE f2.person_id=p.id "
        "  ORDER BY (f2.assigned_by='user') DESC, f2.quality DESC LIMIT 1)) AS cover "
        "FROM person p LEFT JOIN face f ON f.person_id=p.id LEFT JOIN item i ON i.id=f.item_id "
    )

    def people(self, include_hidden: bool = False) -> list[PersonSummary]:
        where = "" if include_hidden else "WHERE p.hidden=0 AND p.kind='person' "
        rows = self._read().execute(
            self._PERSON_SQL + where + "GROUP BY p.id ORDER BY (p.name='') ASC, faces DESC, p.id"
        ).fetchall()
        return [PersonSummary(*r) for r in rows]

    def person(self, person_id: int) -> PersonSummary | None:
        r = self._read().execute(self._PERSON_SQL + "WHERE p.id=? GROUP BY p.id", (person_id,)).fetchone()
        return PersonSummary(*r) if r else None

    def person_faces(self, person_id: int, limit: int = 500, offset: int = 0) -> list[FaceRow]:
        rows = self._read().execute(
            "SELECT f.id, i.media_id, f.person_id, f.assigned_by, f.quality FROM face f "
            "JOIN item i ON i.id=f.item_id WHERE f.person_id=? "
            "ORDER BY (f.assigned_by='user') ASC, f.assign_score ASC, f.quality DESC LIMIT ? OFFSET ?",
            (person_id, limit, offset),
        ).fetchall()
        return [FaceRow(*r) for r in rows]

    def ignored_person_id(self) -> int | None:
        r = self._read().execute("SELECT id FROM person WHERE kind='ignored' ORDER BY id LIMIT 1").fetchone()
        return r[0] if r else None

    def face_media(self, face_id: int) -> tuple[int, tuple[float, float, float, float]] | None:
        r = self._read().execute(
            "SELECT i.media_id, f.x1, f.y1, f.x2, f.y2 FROM face f JOIN item i ON i.id=f.item_id WHERE f.id=?",
            (face_id,),
        ).fetchone()
        return (r[0], (r[1], r[2], r[3], r[4])) if r else None

    # -- changes and undo ------------------------------------------------------------------------------

    def apply_change(self, ch: PeopleChange) -> int | None:
        now = iso(self._now())

        def op(c: sqlite3.Connection) -> int | None:
            target = ch.to_person
            created = None
            if ch.new_person_name is not None:
                created = target = c.execute(
                    "INSERT INTO person(name, kind, created_at) VALUES(?, ?, ?)",
                    (ch.new_person_name.strip(), ch.new_person_kind, now),
                ).lastrowid

            # Two kinds of faces: `explicit` ones the user picked (they become confirmed), and `merged` ones
            # that arrive because their whole person is merged in (they keep their confirmed/auto status).
            explicit = list(dict.fromkeys(ch.face_ids))
            picked = set(explicit)
            merged: list[int] = []
            for pid in ch.delete_persons:
                merged += [r[0] for r in c.execute("SELECT id FROM face WHERE person_id=?", (pid,)) if r[0] not in picked]

            before: dict[int, list] = {}
            for chunk in _chunks(explicit + merged):
                for r in c.execute(
                    f"SELECT id, person_id, assigned_by, assign_score FROM face WHERE id IN ({_placeholders(len(chunk))})",
                    chunk,
                ):
                    before[r[0]] = [r[1], r[2], r[3]]

            if merged and target is not None:
                # A face the user said is "not <target>" must not arrive in target with the rest of its person.
                # The specific decision beats the bulk one; such faces become unassigned instead.
                vetoed = set()
                for chunk in _chunks(merged):
                    vetoed |= {r[0] for r in c.execute(
                        f"SELECT face_id FROM face_reject WHERE person_id=? AND face_id IN ({_placeholders(len(chunk))})",
                        (target, *chunk))}
                if vetoed:
                    v = sorted(vetoed)
                    c.execute(f"UPDATE face SET person_id=NULL, assigned_by=NULL, assign_score=NULL "
                              f"WHERE id IN ({_placeholders(len(v))})", v)
                    merged = [f for f in merged if f not in vetoed]
                for chunk in _chunks(merged):
                    c.execute(f"UPDATE face SET person_id=? WHERE id IN ({_placeholders(len(chunk))})", (target, *chunk))

            for chunk in _chunks(explicit):
                ph = _placeholders(len(chunk))
                if ch.unassign:
                    c.execute(f"UPDATE face SET person_id=NULL, assigned_by=NULL, assign_score=NULL WHERE id IN ({ph})", chunk)
                elif target is not None:
                    c.execute(f"UPDATE face SET person_id=?, assigned_by='user', assign_score=NULL WHERE id IN ({ph})",
                              (target, *chunk))
            if target is not None and not ch.unassign:
                for chunk in _chunks(explicit + merged):
                    c.execute(f"DELETE FROM suggestion_face WHERE face_id IN ({_placeholders(len(chunk))})", chunk)

            face_ids = explicit
            rejects = list(ch.rejects)
            deleted_rejects = []
            if target is not None and not ch.unassign and face_ids:
                # The user explicitly put these faces in `target`: that overrides an earlier "not this person".
                for chunk in _chunks(face_ids):
                    ph = _placeholders(len(chunk))
                    deleted_rejects += [tuple(r) for r in c.execute(
                        f"SELECT face_id, person_id FROM face_reject WHERE person_id=? AND face_id IN ({ph})",
                        (target, *chunk))]
                    c.execute(f"DELETE FROM face_reject WHERE person_id=? AND face_id IN ({ph})", (target, *chunk))
            if ch.delete_persons:
                ph = _placeholders(len(ch.delete_persons))
                merged_rejects = [tuple(r) for r in c.execute(
                    f"SELECT face_id, person_id FROM face_reject WHERE person_id IN ({ph})", ch.delete_persons)]
                deleted_rejects += merged_rejects  # cascade-deleted with the person; restored by undo
                if target is not None:  # "not Mom" stays true after Mom absorbs a duplicate...
                    in_target = {r[0] for r in c.execute("SELECT id FROM face WHERE person_id=?", (target,))}
                    # ...except for a face already in Mom: that later, explicit decision wins (found by the
                    # property test: copying the rejection would leave a face inside a person it is "not").
                    rejects += [(f, target) for f, _ in merged_rejects if f not in in_target]
            added_rejects = []
            for f, p in rejects:
                if c.execute("INSERT OR IGNORE INTO face_reject(face_id, person_id) VALUES(?, ?)", (f, p)).rowcount:
                    added_rejects.append((f, p))

            renamed = None
            if ch.rename:
                pid, name = ch.rename
                old = c.execute("SELECT name FROM person WHERE id=?", (pid,)).fetchone()
                renamed = (pid, old[0] if old else "")
                c.execute("UPDATE person SET name=? WHERE id=?", (name.strip(), pid))

            deleted_persons = []
            for pid in ch.delete_persons:
                row = c.execute("SELECT id, name, kind, hidden, created_at FROM person WHERE id=?", (pid,)).fetchone()
                if row:
                    deleted_persons.append(list(row))
                    c.execute("DELETE FROM person WHERE id=?", (pid,))

            for sid in ch.clear_suggestions:
                c.execute("DELETE FROM suggestion WHERE id=?", (sid,))

            payload = {
                "faces": before, "created": created, "deleted_persons": deleted_persons,
                "deleted_rejects": deleted_rejects, "added_rejects": added_rejects, "renamed": renamed,
            }
            c.execute(
                "INSERT INTO people_event(action, person_id, payload_json, at) VALUES(?, ?, ?, ?)",
                (ch.action, target, json.dumps(payload), now),
            )
            return target

        return self._write(op)

    def undo_last(self) -> str | None:
        def op(c: sqlite3.Connection) -> str | None:
            ev = c.execute(
                "SELECT id, action, payload_json FROM people_event WHERE undone=0 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if ev is None:
                return None
            p = json.loads(ev[2])
            for pid, name, kind, hidden, created_at in p["deleted_persons"]:
                c.execute("INSERT OR IGNORE INTO person(id, name, kind, hidden, created_at) VALUES(?,?,?,?,?)",
                          (pid, name, kind, hidden, created_at))
            for face_id, (person_id, assigned_by, score) in p["faces"].items():
                c.execute("UPDATE face SET person_id=?, assigned_by=?, assign_score=? WHERE id=?",
                          (person_id, assigned_by, score, int(face_id)))
            for f, pid in p["added_rejects"]:
                c.execute("DELETE FROM face_reject WHERE face_id=? AND person_id=?", (f, pid))
            for f, pid in p["deleted_rejects"]:
                c.execute("INSERT OR IGNORE INTO face_reject(face_id, person_id) VALUES(?, ?)", (f, pid))
            if p["renamed"]:
                pid, old = p["renamed"]
                c.execute("UPDATE person SET name=? WHERE id=?", (old, pid))
            if p["created"] is not None:
                c.execute("DELETE FROM person WHERE id=?", (p["created"],))
            c.execute("UPDATE people_event SET undone=1 WHERE id=?", (ev[0],))
            return ev[1]

        return self._write(op)

    def forget_person(self, person_id: int, model_id: str) -> list[int]:
        def op(c: sqlite3.Connection) -> list[int]:
            ids = [r[0] for r in c.execute("SELECT id FROM face WHERE person_id=?", (person_id,))]
            for chunk in _chunks(ids):
                ph = _placeholders(len(chunk))
                c.execute(
                    f"INSERT INTO suppressed_face(model_id, vector) "
                    f"SELECT model_id, vector FROM face_embedding WHERE face_id IN ({ph})",
                    chunk,
                )
                c.execute(f"DELETE FROM face WHERE id IN ({ph})", chunk)
            c.execute("DELETE FROM person WHERE id=?", (person_id,))
            # Undo history must not be able to resurrect forgotten faces.
            c.execute("UPDATE people_event SET undone=1 WHERE undone=0")
            return ids

        return self._write(op)
