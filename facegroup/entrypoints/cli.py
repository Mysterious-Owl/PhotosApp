"""Command line. Parses input, calls services, prints results. No business logic here."""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

import psutil
import typer

from facegroup.bootstrap import build, setup_logging
from facegroup.config import load_settings
from facegroup.domain.entities import Stage

app = typer.Typer(add_completion=False, no_args_is_help=True, help="FaceGroup: local face grouping and search.")


class _Progress:
    def __init__(self) -> None:
        self._last = 0.0
        self._start = time.monotonic()

    def __call__(self, stage: Stage, done: int, failed: int) -> None:
        now = time.monotonic()
        if now - self._last < 0.5:
            return
        self._last = now
        rate = done / max(now - self._start, 1e-6)
        sys.stdout.write(f"\r  {stage.value:<7} {done:>7} done  {failed:>5} failed  {rate:6.1f}/s ")
        sys.stdout.flush()

    def reset(self) -> None:
        self._start = time.monotonic()
        self._last = 0.0
        sys.stdout.write("\n")


def _stop_flag() -> threading.Event:
    stop = threading.Event()

    def handler(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        sys.stdout.write("\n  stopping after the current batch (Ctrl+C again to abort)...\n")

    signal.signal(signal.SIGINT, handler)
    return stop


@app.command()
def scan(
    folder: Path = typer.Argument(None, help="Add this folder to the library, then scan. Omit to scan all."),
    fetch_online_only: bool = typer.Option(False, help="Also read OneDrive online-only files (downloads them)."),
    index: bool = typer.Option(True, help="Run derive and faces after scanning."),
) -> None:
    """Scan library folders for new, changed, moved and deleted photos, then index them."""
    settings = load_settings()
    setup_logging(settings)
    stop = _stop_flag()
    progress = _Progress()
    started = time.monotonic()
    a = build(settings, progress=progress, should_stop=stop.is_set)
    try:
        recovered = a.indexing.recover()
        if recovered:
            typer.echo(f"Resumed {recovered} job(s) left running by a previous run.")
        roots = [a.library.add_root(folder, fetch_online_only)] if folder else a.library.roots()
        if not roots:
            typer.echo("No library folders yet. Run: facegroup scan <folder>")
            raise typer.Exit(1)
        for root in roots:
            r = a.library.scan(root)
            typer.echo(
                f"{root.path}\n  {r.seen} files: {r.new} new, {r.changed} changed, {r.moved} moved, "
                f"{r.unchanged} unchanged, {r.online_only} online-only (skipped), {r.missing} missing, "
                f"{r.removed} removed"
            )
            for e in r.errors:
                typer.echo(f"  ! {e}")
        if index and not stop.is_set():
            for report in (a.indexing.run_derive(), a.indexing.run_faces()):
                progress.reset()
                typer.echo(f"  {report.stage.value}: {report.done} done, {report.failed} failed")
                for e in report.errors[:5]:
                    typer.echo(f"    ! {e}")
        proc = psutil.Process()
        peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
        typer.echo(f"Finished in {time.monotonic() - started:.1f}s, peak memory {peak / 2**30:.2f} GB")
    finally:
        a.close()


@app.command()
def status() -> None:
    """Show library and job counts."""
    a = build(load_settings(), face_model_id="-", analyzer_factory=lambda: None)
    try:
        s = a.store.status()
        typer.echo("Library folders:")
        for r in a.library.roots():
            typer.echo(f"  {r.path}")
        typer.echo("Files:  " + (", ".join(f"{k} {v}" for k, v in sorted(s["media"].items())) or "none"))
        typer.echo("Jobs:   " + (", ".join(f"{k} {v}" for k, v in sorted(s["jobs"].items())) or "none"))
        typer.echo(f"Items:  {s['counts']['items']}   Faces: {s['counts']['faces']}")
        dead = a.store.dead_jobs(10)
        if dead:
            typer.echo("Failed files (first 10):")
            for path, stage, error in dead:
                typer.echo(f"  [{stage}] {path}: {error}")
    finally:
        a.close()


@app.command()
def retry() -> None:
    """Re-queue every failed job."""
    a = build(load_settings(), face_model_id="-", analyzer_factory=lambda: None)
    try:
        typer.echo(f"Re-queued {a.store.retry_dead()} job(s). Run: facegroup scan")
    finally:
        a.close()


@app.command("download-models")
def download_models() -> None:
    """Download the face model now instead of on first scan."""
    from facegroup.adapters.models.buffalo_onnx import ensure_buffalo_l

    settings = load_settings()

    def show(done: int, total: int) -> None:
        sys.stdout.write(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MB")
        sys.stdout.flush()

    path = ensure_buffalo_l(settings.models_dir, progress=show)
    typer.echo(f"\nModels ready in {path}")


if __name__ == "__main__":
    app()
