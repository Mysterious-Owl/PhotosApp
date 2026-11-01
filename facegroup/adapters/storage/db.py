"""SQLite connection setup, migrations and the synced-folder guard."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
_MIGRATION_NAME = re.compile(r"^(\d{4})_.+\.sql$")


class SyncedFolderError(RuntimeError):
    pass


def synced_roots() -> list[Path]:
    """Folders a sync client manages. A SQLite database inside one can be corrupted or duplicated."""
    names = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
    return [Path(os.environ[n]).resolve() for n in names if os.environ.get(n)]


def assert_not_synced(path: Path) -> None:
    resolved = path.resolve()
    for root in synced_roots():
        if resolved == root or root in resolved.parents:
            raise SyncedFolderError(
                f"Refusing to keep the database in a synced folder ({root}). "
                "Move the project out of OneDrive, or set FACEGROUP_DATA_DIR to a local folder."
            )


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    # check_same_thread=False only so close() can run from any thread; each connection is used by one thread.
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None, check_same_thread=False)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    if read_only:
        conn.execute("PRAGMA query_only=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply numbered migrations newer than PRAGMA user_version. Returns the resulting version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    files = sorted(
        (int(m.group(1)), p)
        for p in MIGRATIONS_DIR.iterdir()
        if (m := _MIGRATION_NAME.match(p.name))
    )
    for version, path in files:
        if version <= current:
            continue
        script = path.read_text(encoding="utf-8")
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version={version};\nCOMMIT;")
        current = version
    return current
