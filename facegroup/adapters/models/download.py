"""Model downloads: retried, written to a .part file first, so an interrupted download never looks complete."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Raised with a message meant for the user, not a stack trace."""


def download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None,
             attempts: int = 3, timeout: float = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            log.info("downloading %s (attempt %d)", url, attempt)
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            if total and done != total:
                raise OSError(f"incomplete download ({done} of {total} bytes)")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            log.warning("download failed (attempt %d of %d): %s", attempt, attempts, exc)
            part.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise ModelDownloadError(
        f"Could not download {dest.name} after {attempts} tries ({type(last_error).__name__}: {last_error}). "
        "Check the internet connection and try again; nothing was lost."
    )
