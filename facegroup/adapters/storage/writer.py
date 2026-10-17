"""The single DB writer: one thread owns the only write connection; everyone else submits work to it."""

from __future__ import annotations

import queue
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import TypeVar

from facegroup.adapters.storage.db import connect

T = TypeVar("T")

_STOP = object()


class DbWriter:
    def __init__(self, db_path: Path) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(db_path,), name="db-writer", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    def submit(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run fn(conn) inside one IMMEDIATE transaction on the writer thread and return its result."""
        future: Future[T] = Future()
        self._queue.put((fn, future))
        return future.result()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put((_STOP, None))
            self._thread.join()

    def _run(self, db_path: Path) -> None:
        try:
            conn = connect(db_path)
        except BaseException as exc:  # surface connection errors to the constructor
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                fn, future = self._queue.get()
                if fn is _STOP:
                    break
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    result = fn(conn)
                    conn.execute("COMMIT")
                except BaseException as exc:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    future.set_exception(exc)
                else:
                    future.set_result(result)
        finally:
            conn.close()
