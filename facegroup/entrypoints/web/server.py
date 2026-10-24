"""Local web UI. Thin: parse the request, call one service method, render. No business logic here.

Security (docs/ARCHITECTURE.md §8): bound to 127.0.0.1; Host header allowlisted (with port) against DNS
rebinding; a per-launch token, exchanged once for an HttpOnly SameSite=Strict cookie, is required on every
request; cross-origin POSTs are refused; images are served by id only.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from facegroup.ports.search import SearchFilters

HERE = Path(__file__).parent
COOKIE = "facegroup_session"


def _ids(values: list[str]) -> list[int]:
    return [int(v) for v in values if str(v).isdigit()]


def _safe_next(url: str | None, default: str = "/people") -> str:
    """Only same-site relative paths, so a crafted form cannot redirect elsewhere."""
    if not url or not url.startswith("/") or url.startswith("//"):
        return default
    return url


def create_app(app, scheduler, token: str, port: int) -> FastAPI:
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(directory=HERE / "templates")
    web.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    @web.middleware("http")
    async def guard(request: Request, call_next):
        if request.headers.get("host", "") not in allowed_hosts:
            return Response("Bad host", status_code=400)
        if request.method not in ("GET", "HEAD"):
            origin = request.headers.get("origin")
            if origin and origin not in allowed_origins:
                return Response("Cross-origin request refused", status_code=403)
        if request.url.path == "/login" or request.url.path.startswith("/static/"):
            return await call_next(request)
        if not hmac.compare_digest(request.cookies.get(COOKIE, ""), token):
            return HTMLResponse(
                "<h1>FaceGroup is locked</h1><p>Open it from the link printed in the terminal "
                "(<code>facegroup ui</code>).</p>", status_code=401)
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def page(request: Request, name: str, **ctx):
        ctx.update(status=scheduler.status, request=request)
        return templates.TemplateResponse(request, name, ctx)

    def back(request: Request, next_url: str | None = None, default: str = "/people"):
        return RedirectResponse(_safe_next(next_url, default), status_code=303)

    # -- session ----------------------------------------------------------------------------------------

    @web.get("/login")
    def login(t: str = ""):
        if not hmac.compare_digest(t, token):
            return HTMLResponse("<h1>Invalid or expired link</h1>", status_code=401)
        r = RedirectResponse("/people", status_code=303)
        r.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/")
        return r

    @web.get("/")
    def home():
        return RedirectResponse("/people", status_code=303)

    # -- pages ------------------------------------------------------------------------------------------

    @web.get("/people", response_class=HTMLResponse)
    def people_page(request: Request):
        return page(request, "people.html", people=app.people.people(),
                    suggestions=app.people.suggestions(sample=6), roots=app.library.roots(),
                    counts=app.store.status())

    @web.get("/suggestions/{sid}", response_class=HTMLResponse)
    def suggestion_page(request: Request, sid: int):
        faces = app.people.suggestion_faces(sid)
        if not faces:
            return RedirectResponse("/people", status_code=303)
        return page(request, "suggestion.html", sid=sid, faces=faces, people=app.people.people())

    @web.get("/people/{pid}", response_class=HTMLResponse)
    def person_page(request: Request, pid: int):
        person = app.people.person(pid)
        if person is None:
            return RedirectResponse("/people", status_code=303)
        others = [p for p in app.people.people() if p.id != pid]
        return page(request, "person.html", person=person, faces=app.people.person_faces(pid), others=others)

    # -- suggestion actions -----------------------------------------------------------------------------

    @web.post("/suggestions/{sid}/accept")
    async def accept(request: Request, sid: int):
        form = await request.form()
        if form.get("all"):  # quick save from the overview: every face in the group
            exclude: set[int] = set()
        else:  # review page: unticked faces are left out
            include = set(_ids(form.getlist("face")))
            exclude = {f for f in app.people.suggestion_faces(sid) if f not in include}
        target = form.get("into", "")
        if form.get("use_into") and str(target).isdigit():
            pid = app.people.accept_into(sid, int(target), exclude)
        else:
            pid = app.people.accept(sid, str(form.get("name", "")), exclude)
        return back(request, form.get("next") or f"/people#p{pid}")

    @web.post("/suggestions/{sid}/ignore")
    def ignore(request: Request, sid: int, next: str = Form("/people")):
        app.people.not_a_person(sid)
        return back(request, next)

    # -- person actions ---------------------------------------------------------------------------------

    @web.post("/people/{pid}/rename")
    def rename(request: Request, pid: int, name: str = Form("")):
        app.people.rename(pid, name)
        return back(request, f"/people/{pid}")

    @web.post("/people/{pid}/merge")
    def merge(request: Request, pid: int, into: int = Form(...)):
        app.people.merge(pid, into)
        return back(request, f"/people/{into}")

    @web.post("/people/{pid}/forget")
    def forget(request: Request, pid: int, confirm: str = Form("")):
        if confirm != "forget":
            return back(request, f"/people/{pid}")
        app.people.forget(pid)
        return back(request, "/people")

    @web.post("/people/{pid}/faces")
    async def face_action(request: Request, pid: int):
        form = await request.form()
        faces = _ids(form.getlist("face"))
        action = form.get("action", "")
        target = form.get("target", "")
        if faces:
            if action == "confirm":
                app.people.confirm(faces, pid)
            elif action == "reject":
                app.people.reject(faces, pid)
            elif action == "move" and str(target).isdigit():
                app.people.move(faces, int(target))
            elif action == "split":
                new = app.people.split(faces, str(form.get("name", "")))
                return back(request, f"/people/{new}")
            elif action == "ignore":
                app.people.not_a_person(face_ids=faces)
        return back(request, f"/people/{pid}")

    @web.post("/undo")
    def undo(request: Request, next: str = Form("/people")):
        app.people.undo()
        return back(request, next)

    # -- search -----------------------------------------------------------------------------------------

    @web.get("/search", response_class=HTMLResponse)
    def search_page(request: Request, q: str = "", person: str = "", date_from: str = "", date_to: str = "",
                    folder: str = "", similar: int | None = None, n: int = 60):
        root_id, folder_path = None, None
        if ":" in folder and folder.split(":", 1)[0].isdigit():
            root_id, folder_path = int(folder.split(":", 1)[0]), folder.split(":", 1)[1]
        filters = SearchFilters(
            person_ids=tuple(_ids(person.split(","))) if person else (),
            date_from=date_from or None, date_to=date_to or None, root_id=root_id, folder=folder_path)
        searched = bool(q.strip() or person or date_from or date_to or folder or similar)
        result = None
        if similar is not None:
            result = app.search.similar(similar, filters)
        elif searched:
            result = app.search.search(q, filters)
        done, total = app.search_store.embedding_coverage(app.search.model_id) if app.search.model_id else (0, 0)
        shown = max(12, min(n, 200))
        more_url = None
        if result and len(result.hits) > shown:
            more = request.url.include_query_params(n=shown + 60)
            more_url = f"{more.path}?{more.query}"
        return page(request, "search.html", q=q, result=result, shown=shown, more_url=more_url,
                    people=app.people.people(),
                    folders=app.search_store.top_folders(), sel_person=person, date_from=date_from,
                    date_to=date_to, folder=folder, similar=similar, coverage=(done, total),
                    similar_label=app.media.media_label(similar).replace("\\", "/").split("/")[-1] if similar else "")

    @web.get("/photos/{mid}", response_class=HTMLResponse)
    def photo_page(request: Request, mid: int):
        info = app.search_store.photo_info(mid)
        if info is None:
            return RedirectResponse("/search", status_code=303)
        return page(request, "photo.html", info=info, people=app.people.people())

    # -- library ----------------------------------------------------------------------------------------

    @web.post("/roots")
    def add_root(request: Request, path: str = Form(...)):
        try:
            app.library.add_root(Path(path.strip().strip('"')))
        except (NotADirectoryError, OSError) as exc:
            return page(request, "error.html", message=str(exc))
        scheduler.request_scan()
        return back(request, "/people")

    @web.post("/scan")
    def scan(request: Request, next: str = Form("/people")):
        scheduler.request_scan()
        return back(request, next)

    @web.get("/events")
    async def events(request: Request):
        async def stream():
            last = -1
            while not await request.is_disconnected():
                s = scheduler.status
                if s["version"] != last:
                    last = s["version"]
                    yield f"data: {json.dumps(s)}\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # -- images (by id only) ----------------------------------------------------------------------------

    cache_headers = {"Cache-Control": "private, max-age=3600"}

    @web.get("/faces/{fid}.jpg")
    def face_image(fid: int):
        path = app.media.face_crop(fid)
        return FileResponse(path, media_type="image/jpeg", headers=cache_headers) if path else Response(status_code=404)

    @web.get("/media/{mid}/thumb.jpg")
    def thumb(mid: int):
        path = app.media.thumbnail(mid)
        return FileResponse(path, media_type="image/jpeg", headers=cache_headers) if path else Response(status_code=404)

    @web.get("/media/{mid}/photo.jpg")
    def photo(mid: int):
        data = app.media.photo_jpeg(mid)
        return Response(data, media_type="image/jpeg", headers=cache_headers) if data else Response(status_code=404)

    return web
