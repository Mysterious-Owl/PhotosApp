import threading

import pytest

from facegroup.adapters.instance_lock import AlreadyRunningError, InstanceLock
from facegroup.adapters.storage.db import SyncedFolderError, assert_not_synced, connect, migrate
from facegroup.adapters.storage.writer import DbWriter


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "x.db")
    assert migrate(conn) == 1
    assert migrate(conn) == 1
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"root", "media", "item", "job", "face", "face_embedding", "person", "settings"} <= tables
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_refuses_database_in_synced_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("OneDrive", str(tmp_path))
    with pytest.raises(SyncedFolderError):
        assert_not_synced(tmp_path / "data")
    assert_not_synced(tmp_path.parent / "elsewhere")  # outside: fine


def test_writer_commits_and_rolls_back(tmp_path):
    db = tmp_path / "x.db"
    migrate(connect(db))
    w = DbWriter(db)
    try:
        w.submit(lambda c: c.execute("INSERT INTO settings VALUES('a', '1')"))
        with pytest.raises(ValueError):
            def bad(c):
                c.execute("INSERT INTO settings VALUES('b', '2')")
                raise ValueError("boom")
            w.submit(bad)
        rows = connect(db).execute("SELECT key FROM settings").fetchall()
        assert [r[0] for r in rows] == ["a"]
    finally:
        w.close()


def test_writer_serialises_concurrent_submitters(tmp_path):
    db = tmp_path / "x.db"
    migrate(connect(db))
    w = DbWriter(db)
    try:
        def worker(n):
            for i in range(50):
                w.submit(lambda c, k=f"{n}-{i}": c.execute("INSERT INTO settings VALUES(?, 'v')", (k,)))
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert connect(db).execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 200
    finally:
        w.close()


def test_instance_lock_is_exclusive(tmp_path):
    first = InstanceLock(tmp_path / "facegroup.lock")
    first.acquire()
    try:
        with pytest.raises(AlreadyRunningError):
            InstanceLock(tmp_path / "facegroup.lock").acquire()
    finally:
        first.release()
    again = InstanceLock(tmp_path / "facegroup.lock")
    again.acquire()
    again.release()
