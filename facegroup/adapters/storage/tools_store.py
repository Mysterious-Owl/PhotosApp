"""SQLite storage for the file tools: the latest duplicate scan, folder preferences, choices and the move log."""

from __future__ import annotations

import json
import sqlite3

from facegroup.adapters.storage.base import JSON_IDS, SqliteBase, iso
from facegroup.domain.files import Copy


class SqliteToolsStore(SqliteBase):
    # -- duplicate scan -----------------------------------------------------------------------------------

    def save_scan(self, roots: list[str], started_at: str, groups: list[tuple[str, int, list[Copy]]],
                  files: int, total_bytes: int, cloud_only: int, errors: list[str]) -> int:
        """Replace the previous scan with this one, in one transaction."""
        def op(c: sqlite3.Connection) -> int:
            c.execute("DELETE FROM dup_scan")
            scan = c.execute(
                "INSERT INTO dup_scan(roots, started_at, finished_at, files, bytes, cloud_only, errors) "
                "VALUES(?,?,?,?,?,?,?)",
                (json.dumps(roots), started_at, iso(self._now()), files, total_bytes, cloud_only,
                 json.dumps(errors[:200])),
            ).lastrowid
            for digest, size, copies in groups:
                gid = c.execute("INSERT INTO dup_group(scan_id, hash, size) VALUES(?,?,?)",
                                (scan, digest, size)).lastrowid
                c.executemany("INSERT OR IGNORE INTO dup_file(group_id, path, mtime, size) VALUES(?,?,?,?)",
                              [(gid, cp.path, cp.mtime, cp.size) for cp in copies])
            return scan

        return self._write(op)

    def latest_scan(self) -> dict | None:
        row = self._read().execute("SELECT * FROM dup_scan ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        d = dict(row)
        d["roots"], d["errors"] = json.loads(d["roots"]), json.loads(d["errors"])
        return d

    def dup_groups(self) -> list[tuple[int, str, list[Copy]]]:
        """(group id, hash, copies) for every group in the latest scan that still has two or more copies."""
        rows = self._read().execute(
            "SELECT g.id, g.hash, COALESCE(f.size, g.size), f.path, f.mtime FROM dup_group g "
            "JOIN dup_file f ON f.group_id=g.id "
            "WHERE g.scan_id=(SELECT MAX(id) FROM dup_scan) ORDER BY g.size DESC, g.id, f.path"
        ).fetchall()
        groups: dict[int, tuple[str, list[Copy]]] = {}
        for gid, digest, size, path, mtime in rows:
            groups.setdefault(gid, (digest, []))[1].append(Copy(path, size, mtime))
        return [(gid, digest, copies) for gid, (digest, copies) in groups.items() if len(copies) >= 2]

    def forget_paths(self, paths: list[str]) -> None:
        """Copies that were moved away no longer count as duplicates in the latest scan."""
        if paths:
            self._write(lambda c: c.execute(
                f"DELETE FROM dup_file WHERE path IN {JSON_IDS}", (json.dumps(paths),)))

    # -- preferences and choices -------------------------------------------------------------------------

    def folder_prefs(self) -> dict[tuple[str, str], str]:
        return {(a, b): keep for a, b, keep in self._read().execute("SELECT folder_a, folder_b, keep FROM folder_pref")}

    def set_folder_pref(self, pair: tuple[str, str], keep: str | None) -> None:
        a, b = pair
        if keep is None:
            self._write(lambda c: c.execute("DELETE FROM folder_pref WHERE folder_a=? AND folder_b=?", (a, b)))
            return
        self._write(lambda c: c.execute(
            "INSERT INTO folder_pref(folder_a, folder_b, keep, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(folder_a, folder_b) DO UPDATE SET keep=excluded.keep", (a, b, keep, iso(self._now()))))

    def dup_choices(self) -> dict[str, str]:
        return dict(self._read().execute("SELECT hash, keep FROM dup_choice").fetchall())

    def set_dup_choice(self, digest: str, keep: str | None) -> None:
        if keep is None:
            self._write(lambda c: c.execute("DELETE FROM dup_choice WHERE hash=?", (digest,)))
        else:
            self._write(lambda c: c.execute(
                "INSERT INTO dup_choice(hash, keep) VALUES(?,?) ON CONFLICT(hash) DO UPDATE SET keep=excluded.keep",
                (digest, keep)))

    # -- move log -----------------------------------------------------------------------------------------

    def start_batch(self, kind: str) -> int:
        return self._write(lambda c: c.execute(
            "INSERT INTO removal_batch(created_at, kind) VALUES(?,?)", (iso(self._now()), kind)).lastrowid)

    def record_removed(self, batch_id: int, original: str, holding: str | None, size: int, kept: str | None) -> None:
        """One row per move, written right after the move itself, so a crash mid-batch loses no record."""
        def op(c: sqlite3.Connection) -> None:
            c.execute("INSERT INTO removed_file(batch_id, original, holding, size, kept) VALUES(?,?,?,?,?)",
                      (batch_id, original, holding, size, kept))
            c.execute("UPDATE removal_batch SET files=files+1, bytes=bytes+? WHERE id=?", (size, batch_id))

        self._write(op)

    def batches(self) -> list[dict]:
        rows = self._read().execute(
            "SELECT b.*, (SELECT COUNT(*) FROM removed_file r WHERE r.batch_id=b.id AND r.restored_at IS NULL) "
            "AS still_removed FROM removal_batch b WHERE b.files > 0 ORDER BY b.id DESC").fetchall()
        return [dict(r) for r in rows]

    def batch_files(self, batch_id: int) -> list[dict]:
        rows = self._read().execute("SELECT * FROM removed_file WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
        return [dict(r) for r in rows]

    def holding_paths(self) -> set[str]:
        """Holding paths still in use (casefolded), so a new move never targets one."""
        rows = self._read().execute("SELECT holding FROM removed_file WHERE holding IS NOT NULL AND restored_at IS NULL")
        return {r[0].casefold() for r in rows}

    def mark_restored(self, file_id: int) -> None:
        now = iso(self._now())
        self._write(lambda c: c.execute("UPDATE removed_file SET restored_at=? WHERE id=?", (now, file_id)))
