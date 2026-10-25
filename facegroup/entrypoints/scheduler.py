"""Scheduler: one background thread that runs scan -> derive -> faces -> people refresh when asked.

The web UI reads `status` (a snapshot dict plus a version number) and streams it to the browser.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from facegroup.domain.entities import Stage

log = logging.getLogger(__name__)

_STAGE_LABEL = {Stage.DERIVE: "Preparing photos", Stage.FACES: "Finding faces", Stage.EMBED: "Learning what is in photos",
                Stage.OCR: "Reading text in screenshots and documents"}


class Scheduler:
    def __init__(self) -> None:
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {"state": "idle", "message": "Ready", "done": 0, "failed": 0, "stage": ""}
        self.version = 0
        self._app = None
        self._thread: threading.Thread | None = None

    # -- wiring ---------------------------------------------------------------------------------------

    def attach(self, app) -> None:
        self._app = app

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=60)

    # -- used by IndexingService ----------------------------------------------------------------------

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def progress(self, stage: Stage, done: int, failed: int) -> None:
        self._set(state="indexing", stage=stage.value, done=done, failed=failed,
                  message=f"{_STAGE_LABEL.get(stage, stage.value)}: {done} done"
                          + (f", {failed} failed" if failed else ""))

    # -- used by the web UI ---------------------------------------------------------------------------

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status, version=self.version)

    @property
    def busy(self) -> bool:
        return self.status["state"] not in ("idle", "error")

    def request_scan(self) -> None:
        self._wake.set()

    # -- the loop -------------------------------------------------------------------------------------

    def _set(self, **fields) -> None:
        with self._lock:
            self._status.update(fields)
            self.version += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            started = time.monotonic()
            try:
                self._run_once()
            except Exception as exc:  # never let the thread die; show the error instead
                log.exception("scheduled run failed")
                self._set(state="error", message=f"Indexing stopped: {type(exc).__name__}: {exc}")
            else:
                if not self._stop.is_set():
                    self._set(state="idle", message=f"Up to date ({time.monotonic() - started:.0f}s)")

    def _run_once(self) -> None:
        a = self._app
        for root in a.library.roots():
            self._set(state="scanning", stage="scan", done=0, failed=0, message=f"Scanning {root.path}")
            a.library.scan(root)
        a.indexing.run_all()
        if self._stop.is_set():
            return
        self._set(state="grouping", stage="organize", message="Sorting screenshots, duplicates and bursts")
        a.organize.refresh()
        if not self._stop.is_set():
            a.indexing.run_ocr()
        self._set(state="grouping", stage="people", message="Grouping faces")
        a.people.refresh()
