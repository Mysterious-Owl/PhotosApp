"""Web routes for the file tools: Duplicate files, Convert HEIC, Remove metadata. Thin, like server.py.

Folders come from the user as typed paths, the Browse button (a Windows folder dialog shown by the server, which
runs on the same PC) or the right-click menu; each is checked to be an existing folder. "Show in Explorer" runs
only for a path that is in the latest scan or move log.
"""

from __future__ import annotations

import os
import string
import subprocess
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from facegroup.adapters.tools.picker import pick_folder
from facegroup.domain.files import folder_key, output_folder
from facegroup.entrypoints.web.common import redirect, safe_next
from facegroup.services.tools import (
    Busy,
    ConvertOptions,
    MetadataOptions,
    run_convert,
    run_metadata_check,
    run_metadata_clean,
)

PAGE_SIZE = 40


def clean_folder(value: str) -> Path | None:
    """A typed or pasted folder path, if it exists. Quotes from "Copy as path" are removed."""
    value = (value or "").strip().strip('"').strip()
    if not value:
        return None
    p = Path(os.path.expandvars(value))
    if len(value) == 2 and value[1] == ":":
        p = Path(value + "\\")
    return p if p.is_absolute() and p.is_dir() else None


def drives() -> list[str]:
    if os.name != "nt":
        return []
    return [f"{d}:\\" for d in string.ascii_uppercase[2:] if os.path.isdir(f"{d}:\\")]


def add_tool_routes(web: FastAPI, app, render) -> None:
    runner, dup = app.tools, app.duplicates

    def places() -> list[str]:
        return [str(r.path) for r in app.library.roots()] + drives()

    def tool_page(request: Request, name: str, kind: str, **ctx):
        job = runner.job if runner.job.kind.startswith(kind) else None
        other_job = runner.job if runner.job.running and job is None else None
        return render(request, name, job=job, other_job=other_job, busy=bool(other_job or (job and job.running)),
                      places=places(), **ctx)

    def tool_form(request: Request, tool: str, folder: str, error: str = ""):
        return tool_page(request, f"{tool}.html", tool, folder=folder, error=error)

    def dupes_page(request: Request, view: str, folder: str = "", error: str = "", **ctx):
        return tool_page(request, "duplicates.html", "dupes", view=view, summary=dup.summary(), folder=folder,
                         show="all", error=error, **ctx)

    def run_job(kind: str, title: str, fn, url: str, on_busy):
        """Start a job and go to `url`; if another job is running, show `on_busy(message)` instead."""
        try:
            runner.start(kind, title, fn)
        except Busy as exc:
            return on_busy(str(exc))
        return redirect(url)

    def folder_job(request: Request, tool: str, folder: str, kind: str, title: str, fn, url: str = ""):
        """run_job on the typed folder: fn(src, runner). Back to the tool's form if it is not a folder."""
        src = clean_folder(folder)
        if src is None:
            return tool_form(request, tool, folder, "Not a folder: " + folder)
        return run_job(kind, title, lambda r: fn(src, r), url or f"/tools/{tool}?folder={src}",
                       lambda error: tool_form(request, tool, folder, error))

    # -- job progress (polled by the job panel while a job runs) -------------------------------------------

    @web.get("/tools/job", response_class=HTMLResponse)
    def job_panel(request: Request):
        job = runner.job
        if not job.running:
            return Response(headers={"HX-Refresh": "true"})  # finished: reload the page to show results
        return render(request, "_job.html", job=job)

    @web.post("/tools/pick-folder")
    def pick(initial: str = Form(""), title: str = Form("Choose a folder")):
        """Shows the Windows folder dialog and returns the chosen folder (empty if cancelled)."""
        return JSONResponse({"path": pick_folder(initial, title[:80]) or ""})

    @web.post("/tools/stop")
    def stop():
        runner.stop()  # the job panel keeps polling and reloads the page when the job has wound down
        return Response(status_code=204)

    # -- Convert HEIC --------------------------------------------------------------------------------------

    @web.get("/tools/convert", response_class=HTMLResponse)
    def convert_page(request: Request, folder: str = ""):
        return tool_form(request, "convert", folder)

    @web.post("/tools/convert", response_class=HTMLResponse)
    def convert(request: Request, folder: str = Form(""), output: str = Form(""), fmt: str = Form("png"),
                copy_others: str = Form(""), recurse: str = Form(""), overwrite: str = Form("")):
        def job(src: Path, r):
            opts = ConvertOptions(src, Path(output_folder(output, str(src), "converted")),
                                  "jpg" if fmt == "jpg" else "png", bool(copy_others), bool(overwrite), bool(recurse))
            return run_convert(opts, r)

        return folder_job(request, "convert", folder, "convert", "Convert HEIC", job, "/tools/convert")

    # -- Remove metadata -----------------------------------------------------------------------------------

    @web.get("/tools/metadata", response_class=HTMLResponse)
    def metadata_page(request: Request, folder: str = ""):
        return tool_form(request, "metadata", folder)

    @web.post("/tools/metadata/check", response_class=HTMLResponse)
    def metadata_check(request: Request, folder: str = Form(""), recurse: str = Form("")):
        return folder_job(request, "metadata", folder, "metadata-check", "Check metadata",
                          lambda src, r: run_metadata_check(src, bool(recurse), r))

    @web.post("/tools/metadata/clean", response_class=HTMLResponse)
    def metadata_clean(request: Request, folder: str = Form(""), output: str = Form(""), keep_icc: str = Form(""),
                       keep_date: str = Form(""), recurse: str = Form(""), overwrite: str = Form("")):
        def job(src: Path, r):
            opts = MetadataOptions(src, Path(output_folder(output, str(src), "clean")), bool(keep_icc),
                                   bool(keep_date), bool(overwrite), bool(recurse))
            return run_metadata_clean(opts, r)

        return folder_job(request, "metadata", folder, "metadata-clean", "Remove metadata", job)

    # -- Duplicate files ---------------------------------------------------------------------------------------

    @web.get("/files/duplicates", response_class=HTMLResponse)
    def duplicates_page(request: Request, view: str = "", folder: str = "", show: str = "all", page_no: int = 0):
        summary = dup.summary()
        view = view or ("overview" if summary and not folder else "scan")  # a folder from the right-click menu: scan
        roots = summary["scan"]["roots"] if summary else []
        ctx: dict = {"view": view, "summary": summary, "folder": folder, "error": "", "show": show,
                     "split": lambda p: split_root(p, roots)}
        if view == "pairs":
            ctx["pairs"] = dup.pairs()
        elif view == "groups":
            wanted = {"suggested": ("suggested",), "converted": ("converted",),
                      "ready": ("choice", "rule", "same_folder")}.get(show)
            decisions = [(g, h, d) for g, h, d in dup.decisions() if wanted is None or d.reason in wanted]
            ctx.update(groups=decisions[page_no * PAGE_SIZE:(page_no + 1) * PAGE_SIZE], page_no=page_no,
                       more=(page_no + 1) * PAGE_SIZE < len(decisions), total=len(decisions))
        elif view == "empty":
            ctx["empty"] = dup.empty_folders()
        elif view == "removed":
            ctx["batches"] = dup.batches()
        return tool_page(request, "duplicates.html", "dupes", **ctx)

    @web.post("/files/duplicates/scan", response_class=HTMLResponse)
    def duplicates_scan(request: Request, folders: str = Form(""), media_only: str = Form(""), min_kb: int = Form(1)):
        roots, bad = [], []
        for line in folders.splitlines():
            if line.strip():
                (roots if clean_folder(line) else bad).append(clean_folder(line) or line.strip())
        if bad or not roots:
            return dupes_page(request, "scan", folders,
                              "Not a folder: " + ", ".join(bad) if bad else "Add at least one folder.")
        return run_job("dupes-scan", "Find duplicate files",
                       lambda r: dup.run_scan(roots, bool(media_only), max(0, min_kb) * 1024, r),
                       "/files/duplicates?view=scan", lambda error: dupes_page(request, "scan", folders, error))

    @web.post("/files/duplicates/prefer")
    def duplicates_prefer(a: str = Form(...), b: str = Form(...), keep: str = Form(""),
                          next: str = Form("/files/duplicates?view=pairs")):
        dup.prefer(a, b, keep or None)
        return redirect(safe_next(next, "/files/duplicates"))

    @web.post("/files/duplicates/choose")
    def duplicates_choose(digest: str = Form(...), keep: str = Form(""),
                          next: str = Form("/files/duplicates?view=groups")):
        dup.choose(digest, keep or None)
        return redirect(safe_next(next, "/files/duplicates"))

    @web.post("/files/duplicates/remove", response_class=HTMLResponse)
    def duplicates_remove(request: Request, confirm: str = Form(""), include_suggested: str = Form(""),
                          include_converted: str = Form("")):
        if confirm != "yes":
            return redirect("/files/duplicates")
        return run_job("dupes-remove", "Move extra copies",
                       lambda r: dup.run_remove(bool(include_suggested), r, include_converted=bool(include_converted)),
                       "/files/duplicates?view=removed", lambda error: dupes_page(request, "overview", error=error))

    @web.post("/files/duplicates/empty")
    def duplicates_empty(confirm: str = Form("")):
        if confirm == "yes":
            dup.remove_empty_folders(dup.empty_folders())
        return redirect("/files/duplicates?view=removed")

    @web.post("/files/duplicates/restore/{batch_id}", response_class=HTMLResponse)
    def duplicates_restore(request: Request, batch_id: int):
        return run_job("dupes-restore", "Restore", lambda r: dup.run_restore(batch_id, r),
                       "/files/duplicates?view=removed",
                       lambda error: dupes_page(request, "removed", error=error, batches=dup.batches()))

    @web.get("/files/duplicates/batch/{batch_id}", response_class=HTMLResponse)
    def duplicates_batch(request: Request, batch_id: int):
        return render(request, "_batch.html", files=dup.batch_files(batch_id))

    @web.post("/files/reveal")
    def reveal(path: str = Form(...)):
        """Open Explorer with the file selected. Only for files FaceGroup listed, never an arbitrary path."""
        known = {folder_key(c.path) for _, _, copies in app.tools_store.dup_groups() for c in copies}
        for b in dup.batches()[:50]:
            known |= {folder_key(f["holding"] or f["original"]) for f in dup.batch_files(b["id"])}
        if folder_key(path) in known and os.name == "nt":
            subprocess.Popen(["explorer.exe", "/select,", os.path.normpath(path)])
        return Response(status_code=204)


def split_root(path: str, roots: list[str]) -> tuple[str, str]:
    """('D:\\Scanned\\', 'Trip\\a.jpg'): the scanned folder a path is under, shown faintly, and the part that
    matters. Long absolute paths are otherwise hard to compare at a glance."""
    key = folder_key(path)
    for root in sorted(roots, key=len, reverse=True):
        r = folder_key(root)
        if key.startswith(r + "\\"):
            cut = len(root.rstrip("\\/")) + 1
            return path[:cut], path[cut:]
    return "", path
