"""Shared pieces of the SQLite stores: connections, timestamps, id lists and vector blobs."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from facegroup.adapters.storage.db import connect
from facegroup.adapters.storage.writer import DbWriter
from facegroup.domain.clock import utc_now


# Id lists go in as one JSON value (`WHERE id IN {JSON_IDS}` with `json_ids(ids)`): tens of thousands of "?"
# would pass SQLite's variable limit, and read connections are query-only, so no temp table either.
JSON_IDS = "(SELECT value FROM json_each(?))"


def json_ids(ids: Iterable) -> str:
    return json.dumps([int(i) for i in ids])


def stack_vectors(blobs: list[bytes], empty_dim: int) -> np.ndarray:
    """float32 blobs as rows of a matrix; (0, empty_dim) when there are none."""
    if not blobs:
        return np.zeros((0, empty_dim), np.float32)
    return np.stack([np.frombuffer(b, np.float32) for b in blobs])


class SqliteBase:
    """Per-thread read-only connections for reads; every write goes through the single DbWriter."""

    def __init__(self, db_path: Path, writer: DbWriter, now: Callable[[], datetime] = utc_now) -> None:
        self._db_path = db_path
        self._writer = writer
        self._now = now
        self._local = threading.local()
        self._all_reads: list[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def _read(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._db_path, read_only=True)
            self._local.conn = conn
            with self._lock:
                self._all_reads.append(conn)
        return conn

    def _write(self, fn):
        return self._writer.submit(fn)

    def close(self) -> None:
        with self._lock:
            for conn in self._all_reads:
                try:
                    conn.close()
                except sqlite3.ProgrammingError:
                    pass  # already closed
            self._all_reads.clear()
        self._local = threading.local()
