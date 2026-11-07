"""Local web UI. Thin: parse the request, call one service method, render. No business logic here.

Security (docs/ARCHITECTURE.md §8): bound to 127.0.0.1; Host header allowlisted (with port) against DNS
rebinding; a per-launch token, exchanged once for an HttpOnly SameSite=Strict cookie, is required on every
request; cross-origin POSTs are refused; images are served by id only.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import hmac
import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from facegroup.domain.files import HOLDING_FOLDER
from facegroup.entrypoints.web.common import redirect, safe_next
from facegroup.entrypoints.web.tools_routes import add_tool_routes
from facegroup.ports.search import SearchFilters
from facegroup.services.organize import CATEGORIES

KIND_LABELS = {"photo": "Photos", "screenshot": "Screenshots", "document": "Documents", "receipt": "Receipts",
               "meme": "Memes", "illustration": "Illustrations & wallpapers"}
KINDS = [("", "Everything"), ("photos", "Photos only (hide clutter)"),
         *[(k, v) for k, v in KIND_LABELS.items() if k != "photo"], ("hidden", "Hidden photos")]

HERE = Path(__file__).parent
COOKIE = "facegroup_session"
CACHE_HEADERS = {"Cache-Control": "private, max-age=3600"}


def _ids(values: list[str]) -> list[int]:
    return [int(v) for v in values if str(v).isdigit()]


def person_label(p) -> str:
    """Unnamed people get a stable id-based label ("Face 12") so they can be told apart and referred to."""
    name = p.get("name") if isinstance(p, dict) else p.name
    pid = p.get("person_id", p.get("id")) if isinstance(p, dict) else p.id
    return name.strip() if name and name.strip() else f"Face {pid}"


def human_size(n: int | None) -> str:
    n = n or 0
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}" if unit == "GB" else f"{n / scale:.0f} {unit}"
    return f"{n} bytes"


def _form_int(form, key: str) -> int | None:
    value = str(form.get(key, ""))
    return int(value) if value.isdigit() else None


def _jpeg_file(path) -> Response:
    return FileResponse(path, media_type="image/jpeg", headers=CACHE_HEADERS) if path else Response(status_code=404)


def create_app(app, scheduler, token: str, port: int) -> FastAPI:
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["label"] = person_label
    templates.env.filters["size"] = human_size
    templates.env.filters["num"] = "{:,}".format
    templates.env.filters["ts"] = lambda t: datetime.fromtimestamp(t).strftime("%d %b %Y") if t else ""
    templates.env.globals["HOLDING_FOLDER"] = HOLDING_FOLDER
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

    def render(request: Request, name: str, **ctx):
        ctx.update(status=scheduler.status, request=request)
        return templates.TemplateResponse(request, name, ctx)

    def back(next_url: str | None = None, default: str = "/people"):
        return redirect(safe_next(next_url, default))

    def media_of(face_ids) -> dict[int, int]:
        return {f: (app.people_store.face_media(f) or (0, None))[0] for f in face_ids}

    # -- session ----------------------------------------------------------------------------------------

    @web.get("/login")
    def login(t: str = "", next: str = "/photos"):
        if not hmac.compare_digest(t, token):
            return HTMLResponse("<h1>Invalid or expired link</h1>", status_code=401)
        r = redirect(safe_next(next, "/photos"))  # e.g. a tool page from the right-click menu
        r.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/")
        return r

    @web.get("/")
    def home():
        return redirect("/photos")

    # -- browse: timeline, memories, events, saved searches ----------------------------------------------

    def browse_filters(everything: str) -> SearchFilters:
        return SearchFilters(kind=None if everything else "photos")  # clutter hidden unless asked

    @web.get("/photos", response_class=HTMLResponse)
    def photos_page(request: Request, page: int = 0, everything: str = "", month: str = ""):
        if not app.library.roots():
            return redirect("/people")  # first run: the add-folder screen
        f = browse_filters(everything)
        if month:
            page = app.browse.month_page(f, month)
        return render(request, "photos.html", tl=app.browse.timeline(f, max(page, 0)),
                      memories=app.browse.memories(f) if page == 0 else [], everything=everything)

    @web.get("/events", response_class=HTMLResponse)
    def events_page(request: Request, everything: str = ""):
        return render(request, "events.html", events=app.browse.events(browse_filters(everything)), everything=everything)

    @web.get("/event", response_class=HTMLResponse)
    def event_page(request: Request, key: str, everything: str = ""):
        event = app.browse.event(browse_filters(everything), key)
        if event is None:
            return redirect("/events")
        return render(request, "event.html", event=event)

    @web.post("/search/save")
    async def save_search(request: Request):
        form = await request.form()
        try:
            app.browse.save(str(form.get("name", "")), {k: str(v) for k, v in form.items()})
        except ValueError as exc:
            return render(request, "error.html", message=str(exc))
        return back("/search?" + app.browse.safe_query(str(form.get("current", ""))))

    @web.post("/search/saved/{sid}/delete")
    def delete_saved(sid: int):
        app.browse.delete(sid)
        return back("/search")

    @web.get("/saved/{sid}")
    def open_saved(sid: int):
        found = next((s for s in app.browse.saved() if s["id"] == sid), None)
        return redirect("/search?" + app.browse.safe_query(found["query"]) if found else "/search")

    # -- pages ------------------------------------------------------------------------------------------

    @web.get("/people", response_class=HTMLResponse)
    def people_page(request: Request):
        return render(request, "people.html", people=app.people.people(), hidden=app.people.hidden(),
                      suggestions=app.people.suggestions(sample=6), roots=app.library.roots(),
                      counts=app.store.status())

    @web.get("/suggestions/{sid}", response_class=HTMLResponse)
    def suggestion_page(request: Request, sid: int):
        faces = app.people.suggestion_faces(sid)
        if not faces:
            return redirect("/people")
        return render(request, "suggestion.html", sid=sid, faces=faces, parts=app.people.parts(faces),
                      people=app.people.people(), media_of=media_of(faces))

    @web.get("/people/{pid}", response_class=HTMLResponse)
    def person_page(request: Request, pid: int):
        person = app.people.person(pid)
        if person is None:
            return redirect("/people")
        others = [p for p in app.people.people() if p.id != pid]
        faces = app.people.person_faces(pid)
        by_id = {f.id: f for f in faces}
        parts = [[by_id[i] for i in part] for part in app.people.parts(list(by_id))]
        return render(request, "person.html", person=person, faces=faces, parts=parts, others=others)

    # -- suggestion actions -----------------------------------------------------------------------------

    @web.post("/suggestions/{sid}/accept")
    async def accept(request: Request, sid: int):
        form = await request.form()
        if form.get("all"):  # quick save from the overview: every face in the group
            exclude: set[int] = set()
        else:  # review page: unticked faces are left out
            include = set(_ids(form.getlist("face")))
            exclude = {f for f in app.people.suggestion_faces(sid) if f not in include}
        into = _form_int(form, "into")
        if form.get("use_into") and into is not None:
            pid = app.people.accept_into(sid, into, exclude)
        else:
            pid = app.people.accept(sid, str(form.get("name", "")), exclude)
        return back(form.get("next") or f"/people#p{pid}")

    @web.post("/suggestions/{sid}/faces")
    async def suggestion_faces_action(request: Request, sid: int):
        """Act on the picked faces only; the rest of the group stays, so a mixed group can be broken up."""
        form = await request.form()
        allowed = set(app.people.suggestion_faces(sid))
        faces = [f for f in _ids(form.getlist("face")) if f in allowed]
        action, target = form.get("action", ""), _form_int(form, "target")
        if faces:
            if action == "new":
                app.people.split(faces, str(form.get("name", "")))
            elif action == "add" and target is not None:
                app.people.move(faces, target)
            elif action == "ignore":
                app.people.not_a_person(face_ids=faces)
        left = app.people.suggestion_faces(sid)
        return back(f"/suggestions/{sid}" if left else "/people")

    # -- person actions ---------------------------------------------------------------------------------

    @web.post("/people/{pid}/rename")
    def rename(pid: int, name: str = Form("")):
        app.people.rename(pid, name)
        return back(f"/people/{pid}")

    @web.post("/people/{pid}/merge")
    def merge(pid: int, into: int = Form(...)):
        app.people.merge(pid, into)
        return back(f"/people/{into}")

    @web.post("/people/{pid}/forget")
    def forget(pid: int, confirm: str = Form("")):
        if confirm != "forget":
            return back(f"/people/{pid}")
        app.people.forget(pid)
        return back("/people")

    @web.post("/people/{pid}/faces")
    async def face_action(request: Request, pid: int):
        form = await request.form()
        faces = _ids(form.getlist("face"))
        action, target = form.get("action", ""), _form_int(form, "target")
        if faces:
            if action == "confirm":
                app.people.confirm(faces, pid)
            elif action == "reject":
                app.people.reject(faces, pid)
            elif action == "move" and target is not None:
                app.people.move(faces, target)
            elif action == "split":
                new = app.people.split(faces, str(form.get("name", "")))
                return back(f"/people/{new}")
            elif action == "ignore":
                app.people.not_a_person(face_ids=faces)
            elif action == "restore":
                app.people.restore(faces)
        return back(f"/people/{pid}")

    @web.post("/people/combine")
    async def combine(request: Request):
        form = await request.form()
        people_ids = _ids(form.getlist("person"))
        suggestion_ids = _ids(form.getlist("suggestion"))
        if len(people_ids) + len(suggestion_ids) < 2:
            return render(request, "error.html", message="Tick at least two people or groups to combine.")
        pid = app.people.combine(people_ids, suggestion_ids, str(form.get("name", "")))
        return back(f"/people/{pid}" if pid else "/people")

    @web.get("/check", response_class=HTMLResponse)
    def check_page(request: Request):
        questions = app.people.questions()
        people = {p.id: p for p in app.people.people()}
        questions = [q for q in questions if q.person_id in people and (q.other_id is None or q.other_id in people)]
        return render(request, "check.html", questions=questions, people=people,
                      media_of=media_of(q.face_id for q in questions if q.face_id),
                      stats=app.people.learning_stats(), min_samples=30)

    @web.post("/check/answer")
    async def check_answer(request: Request):
        form = await request.form()
        kind, yes = form.get("kind"), form.get("answer") == "yes"
        face, person, other, group = (_form_int(form, k) for k in ("face", "person", "other", "suggestion"))
        if kind == "is_person" and face and person:
            app.people.answer_is_person(face, person, yes)
        elif kind == "same_person" and person and other:
            app.people.answer_same_person(person, other, yes)
        elif kind == "group_is" and group and person:
            app.people.answer_group_is(group, person, yes)
        return back("/check")

    @web.post("/undo")
    def undo(next: str = Form("/people")):
        app.people.undo()
        return back(next)

    # -- search -----------------------------------------------------------------------------------------

    @web.get("/search", response_class=HTMLResponse)
    def search_page(request: Request, q: str = "", person: str = "", date_from: str = "", date_to: str = "",
                    folder: str = "", similar: int | None = None, n: int = 60, kind: str = "", fav: str = "",
                    names: str = "1"):
        people = app.people.people()
        root_id, folder_path = None, None
        if ":" in folder and folder.split(":", 1)[0].isdigit():
            root_id, folder_path = int(folder.split(":", 1)[0]), folder.split(":", 1)[1]
        filters = SearchFilters(
            person_ids=tuple(_ids(person.split(","))) if person else (),
            date_from=date_from or None, date_to=date_to or None, root_id=root_id, folder=folder_path,
            kind=kind or None, favorites=bool(fav))
        searched = bool(q.strip() or person or date_from or date_to or folder or similar or kind or fav)
        result = None
        if similar is not None:
            result = app.search.similar(similar, filters)
        elif searched:
            result = app.search.search(q, filters, parse_names=names != "0")
        done, total = app.search_store.embedding_coverage(app.search.model_id) if app.search.model_id else (0, 0)
        shown = max(12, min(n, 200))
        more_url = None
        if result and len(result.hits) > shown:
            more = request.url.include_query_params(n=shown + 60)
            more_url = f"{more.path}?{more.query}"
        literal = request.url.include_query_params(names="0")
        return render(request, "search.html", q=q, result=result, shown=shown, more_url=more_url,
                      saved=app.browse.saved(), current_query=request.url.query,
                      literal_url=f"{literal.path}?{literal.query}",
                      people_by_id={p.id: p for p in people}, people=people,
                      folders=app.search_store.top_folders(), sel_person=person, date_from=date_from,
                      date_to=date_to, folder=folder, similar=similar, coverage=(done, total), kind=kind, fav=fav,
                      kinds=KINDS,
                      similar_label=app.media.media_label(similar).replace("\\", "/").split("/")[-1] if similar else "")

    @web.get("/photos/{mid}", response_class=HTMLResponse)
    def photo_page(request: Request, mid: int, face: int | None = None):
        info = app.search_store.photo_info(mid)
        if info is None:
            return redirect("/search")
        dup_ids = app.organize_store.duplicate_set_of(mid)
        return render(request, "photo.html", info=info, highlight=face, flags=app.organize_store.flags(mid),
                      stack=app.organize_store.stack_of(mid), duplicates=[d for d in dup_ids if d != mid],
                      categories=CATEGORIES)

    @web.post("/photos/{mid}/flags")
    async def photo_flags(request: Request, mid: int):
        form = await request.form()
        if "favorite" in form:
            app.organize_store.set_flags([mid], favorite=form.get("favorite") == "1")
        if "rating" in form:
            r = _form_int(form, "rating")
            app.organize_store.set_flags([mid], rating=r if r is not None and 1 <= r <= 5 else None)
        if "hidden" in form:
            app.organize_store.set_flags([mid], hidden=form.get("hidden") == "1")
        if form.get("category") in CATEGORIES:
            app.organize.set_category(mid, str(form.get("category")))
        return back(f"/photos/{mid}")

    @web.post("/photos/{mid}/stack-top")
    def stack_top(mid: int):
        app.organize_store.set_stack_top(mid)
        return back(f"/photos/{mid}")

    @web.get("/tidy", response_class=HTMLResponse)
    def tidy_page(request: Request):
        return render(request, "tidy.html", summary=app.organize.summary(), sets=app.organize.duplicate_sets()[:200],
                      labels=KIND_LABELS)

    @web.post("/duplicates/keep")
    async def keep_copy(request: Request):
        form = await request.form()
        keeper = _form_int(form, "keeper")
        members = app.organize_store.duplicate_set_of(keeper) if keeper else []
        if keeper and members:
            app.organize.hide_extra_copies(members, keeper)
        return back("/tidy")

    # -- library ----------------------------------------------------------------------------------------

    @web.post("/roots")
    def add_root(request: Request, path: str = Form(...)):
        try:
            app.library.add_root(Path(path.strip().strip('"')))
        except (NotADirectoryError, OSError) as exc:
            return render(request, "error.html", message=str(exc))
        scheduler.request_scan()
        return back("/people")

    @web.post("/scan")
    def scan(next: str = Form("/people")):
        scheduler.request_scan()
        return back(next)

    @web.get("/status/stream")  # not "/events": that is the Events page
    async def status_stream(request: Request):
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

    @web.get("/faces/{fid}.jpg")
    def face_image(fid: int):
        return _jpeg_file(app.media.face_crop(fid))

    @web.get("/media/{mid}/thumb.jpg")
    def thumb(mid: int):
        return _jpeg_file(app.media.thumbnail(mid))

    @web.get("/media/{mid}/photo.jpg")
    def photo(mid: int):
        data = app.media.photo_jpeg(mid)
        return Response(data, media_type="image/jpeg", headers=CACHE_HEADERS) if data else Response(status_code=404)

    add_tool_routes(web, app, render)

    return web
