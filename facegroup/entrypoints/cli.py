"""Command line. Parses input, calls services, prints results. No business logic here."""

from __future__ import annotations

import signal
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import psutil
import typer

from facegroup.bootstrap import build, face_analyzer_factory, setup_logging
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
            for report in a.indexing.run_all():
                progress.reset()
                typer.echo(f"  {report.stage.value}: {report.done} done, {report.failed} failed")
                for e in report.errors[:5]:
                    typer.echo(f"    ! {e}")
            if not stop.is_set():
                o = a.organize.refresh()
                typer.echo(f"  organise: {o['categorised']} categorised, {o['duplicate_sets']} duplicate set(s), "
                           f"{o['stacks']} burst stack(s)")
                r = a.indexing.run_ocr()
                progress.reset()
                typer.echo(f"  {r.stage.value}: {r.done} done, {r.failed} failed")
                changed, groups = a.people.refresh()
                typer.echo(f"  people: {changed} face(s) matched to known people, {groups} suggested group(s)")
        proc = psutil.Process()
        peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
        typer.echo(f"Finished in {time.monotonic() - started:.1f}s, peak memory {peak / 2**30:.2f} GB")
    finally:
        a.close()


def _open_library(with_face_model: bool = False):
    """The library without loading any model, for the quick read-only commands; closed at the end of `with`."""
    settings = load_settings()
    model_id = face_analyzer_factory(settings)[1] if with_face_model else "-"
    return closing(build(settings, face_model_id=model_id, analyzer_factory=lambda: None))


@app.command()
def status() -> None:
    """Show library and job counts."""
    with _open_library() as a:
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


@app.command()
def people() -> None:
    """List people and suggested groups."""
    with _open_library(with_face_model=True) as a:
        for p in a.people.people():
            typer.echo(f"  {p.name or f'Face {p.id}':<30} {p.faces:>5} faces ({p.confirmed} confirmed) in {p.photos} photos")
        s = a.people.suggestions()
        typer.echo(f"{len(s)} suggested group(s): " + ", ".join(str(g.size) for g in s[:20]))
        typer.echo("Open the People screen with: facegroup ui")


@app.command()
def retry() -> None:
    """Re-queue every failed job."""
    with _open_library() as a:
        typer.echo(f"Re-queued {a.store.retry_dead()} job(s). Run: facegroup scan")


@app.command()
def ui(
    port: int = typer.Option(None, help="Port on 127.0.0.1 (default from config: 8765)."),
    scan_on_start: bool = typer.Option(True, "--scan/--no-scan", help="Look for new photos when it starts."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the browser automatically."),
    page: str = typer.Option("photos", help="Page to open: photos, people, duplicates, convert, metadata."),
    folder: str = typer.Option(None, help="Folder to fill in on a tool page (used by the right-click menu)."),
) -> None:
    """Open FaceGroup in its own window. Ctrl+C to quit. If it is already running, opens the page there."""
    import secrets

    import uvicorn

    from facegroup.adapters.instance_lock import AlreadyRunningError
    from facegroup.entrypoints import launch
    from facegroup.entrypoints.scheduler import Scheduler
    from facegroup.entrypoints.web.server import create_app

    settings = load_settings()
    setup_logging(settings)
    port = port or settings.port
    next_path = launch.page_url(page, folder)
    scheduler = Scheduler()
    try:
        a = build(settings, progress=scheduler.progress, should_stop=scheduler.should_stop)
    except AlreadyRunningError:
        session = launch.read_session(settings.data_dir)
        if session is None:
            raise
        launch.open_window(launch.login_url(session["port"], session["token"], next_path))
        typer.echo("FaceGroup is already running: opened it there.")
        return
    try:
        a.indexing.recover()
        scheduler.attach(a)
        scheduler.start()
        threading.Thread(target=a.search.warm_up, name="search-warm-up", daemon=True).start()
        if scan_on_start and a.library.roots():
            scheduler.request_scan()
        token = secrets.token_urlsafe(32)
        url = launch.login_url(port, token, next_path)
        web = create_app(a, scheduler, token, port)
        launch.write_session(settings.data_dir, port, token)
        typer.echo(f"FaceGroup is running. Open this link (it works only on this computer, this session):\n  {url}")
        if open_browser:
            threading.Timer(1.0, lambda: launch.open_window(url)).start()
        uvicorn.run(web, host="127.0.0.1", port=port, log_level="warning")
    finally:
        launch.clear_session(settings.data_dir)
        scheduler.stop()
        a.close()


def _run_tool(title: str, fn) -> dict:
    """Run a file tool in this console with a progress line. They need no database, so FaceGroup may be open."""
    from facegroup.services.tools import ToolRunner

    runner = ToolRunner()
    signal.signal(signal.SIGINT, lambda *_: runner.stop())
    runner.start("cli", title, fn)
    while runner.job.running:
        j = runner.job
        sys.stdout.write(f"\r  {j.message[:60]:<60} {j.done:>7}/{j.total:<7}")
        sys.stdout.flush()
        time.sleep(0.3)
    typer.echo(f"\n{runner.job.message}")
    for e in (runner.job.result.get("errors") or [])[:20]:
        typer.echo(f"  ! {e}")
    return runner.job.result


@app.command()
def convert(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder with HEIC/HEIF photos."),
    jpg: bool = typer.Option(False, "--jpg", help="JPEG instead of PNG (much smaller)."),
    copy_all: bool = typer.Option(False, "--all", help="Also copy every other file, so the new folder is complete."),
    recurse: bool = typer.Option(False, "--subfolders", help="Include subfolders."),
    overwrite: bool = typer.Option(False, help="Replace files already in the output folder."),
    output: Path = typer.Option(None, help="Output folder (default: '<folder> converted' next to it)."),
) -> None:
    """Convert HEIC/HEIF photos to PNG or JPEG copies. Originals are not changed."""
    from facegroup.domain.files import output_folder
    from facegroup.services.tools import ConvertOptions, run_convert

    src = folder.resolve()
    opts = ConvertOptions(src, Path(output_folder(str(output or ""), str(src), "converted")),
                          "jpg" if jpg else "png", copy_all, overwrite, recurse)
    _run_tool("Convert HEIC", lambda r: run_convert(opts, r))


@app.command()
def metadata(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder with pictures."),
    clean: bool = typer.Option(False, "--clean", help="Write clean copies (default: only report what is there)."),
    keep_date: bool = typer.Option(False, help="Keep the date taken in clean copies."),
    output: Path = typer.Option(None, help="Output folder for --clean (default: '<folder> clean' next to it)."),
) -> None:
    """Report the metadata in pictures (location, camera, date), or write copies without it."""
    from facegroup.domain.files import output_folder
    from facegroup.services.tools import MetadataOptions, run_metadata_check, run_metadata_clean

    src = folder.resolve()
    if clean:
        opts = MetadataOptions(src, Path(output_folder(str(output or ""), str(src), "clean")), keep_date=keep_date)
        _run_tool("Remove metadata", lambda r: run_metadata_clean(opts, r))
    else:
        _run_tool("Check metadata", lambda r: run_metadata_check(src, True, r))


@app.command("context-menu")
def context_menu(action: str = typer.Argument("install", help="install or remove")) -> None:
    """Add (or remove) a FaceGroup submenu to Explorer's right-click menu, for the current user only."""
    from facegroup.entrypoints import context_menu as menu

    if action == "remove":
        menu.remove()
        typer.echo("Removed the FaceGroup right-click menu.")
        return
    launcher = Path(__file__).resolve().parents[2] / "FaceGroup.cmd"
    if not launcher.exists():
        raise typer.BadParameter(f"{launcher} not found; run this from the FaceGroup project folder.")
    n = menu.install(launcher)
    typer.echo(f"Added {n} right-click items (Windows 11: under 'Show more options'). Remove with: "
               "facegroup context-menu remove")


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


def main() -> None:
    from facegroup.adapters.models.download import ModelDownloadError

    try:
        app()
    except ModelDownloadError as exc:
        typer.echo(f"\n{exc}", err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
