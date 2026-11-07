"""The tool pages through the web app, on synthetic files in temporary folders."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from facegroup.entrypoints.scheduler import Scheduler
from facegroup.entrypoints.web.server import create_app
from tests.conftest import make_photo
from tests.unit.test_web import BASE, PORT, TOKEN, login


@pytest.fixture
def web(make_app, tmp_path, monkeypatch):
    from facegroup.services import tools
    holding = tmp_path / "FaceGroup removed"
    monkeypatch.setattr(tools, "holding_path",
                        lambda p, taken=None: str(holding / Path(p).relative_to(tmp_path)))
    app = make_app()
    client = TestClient(create_app(app, Scheduler(), TOKEN, PORT), base_url=BASE, follow_redirects=False)
    login(client)
    return app, client


def test_every_page_renders_with_the_sidebar(web):
    _, client = web
    for url in ("/photos", "/people", "/events", "/search", "/tidy", "/check", "/files/duplicates",
                "/files/duplicates?view=removed", "/tools/convert", "/tools/metadata"):
        r = client.get(url)
        if r.status_code == 303:  # first run: Photos sends you to add a folder
            continue
        assert r.status_code == 200, url
        assert 'class="side"' in r.text and "Duplicate files" in r.text
    assert 'aria-current="page"' in client.get("/tools/convert").text


def test_duplicates_through_the_ui(web, tmp_path):
    app, client = web
    lib = tmp_path / "lib"
    make_photo(lib / "Photos" / "a.jpg", color=(1, 2, 3))
    (lib / "Backup").mkdir()
    shutil.copy2(lib / "Photos" / "a.jpg", lib / "Backup" / "a.jpg")
    r = client.post("/files/duplicates/scan", data={"folders": f'"{lib}"\n', "media_only": "1", "min_kb": "0"})
    assert r.status_code == 303
    app.tools.wait()
    overview = client.get("/files/duplicates").text
    assert "sets of identical files" in overview and "a decision" in overview
    pairs = client.get("/files/duplicates?view=pairs").text
    assert "Keep copies here" in pairs
    client.post("/files/duplicates/prefer", data={"a": str(lib / "Backup"), "b": str(lib / "Photos"),
                                                  "keep": str(lib / "Photos")})
    assert "folder rule" in client.get("/files/duplicates?view=groups").text
    assert client.post("/files/duplicates/remove", data={}).status_code == 303  # no confirmation: nothing moves
    assert (lib / "Backup" / "a.jpg").exists()
    client.post("/files/duplicates/remove", data={"confirm": "yes"})
    app.tools.wait()
    assert not (lib / "Backup" / "a.jpg").exists() and (lib / "Photos" / "a.jpg").exists()
    moved = client.get("/files/duplicates?view=removed").text
    assert "Restore" in moved
    batch = app.duplicates.batches()[0]["id"]
    assert "Backup" in client.get(f"/files/duplicates/batch/{batch}").text
    client.post(f"/files/duplicates/restore/{batch}")
    app.tools.wait()
    assert (lib / "Backup" / "a.jpg").exists()


def test_bad_folder_is_explained(web):
    _, client = web
    r = client.post("/tools/convert", data={"folder": r"Z:\no\such\folder"})
    assert r.status_code == 200 and "Not a folder" in r.text
    r = client.post("/files/duplicates/scan", data={"folders": "nope"})
    assert "Not a folder" in r.text


def test_convert_and_metadata_jobs(web, tmp_path):
    app, client = web
    src = tmp_path / "pics"
    make_photo(src / "a.jpg")
    client.post("/tools/metadata/check", data={"folder": str(src), "recurse": "1"})
    app.tools.wait()
    assert "pictures checked" in client.get("/tools/metadata").text
    client.post("/tools/metadata/clean", data={"folder": str(src), "keep_icc": "1"})
    app.tools.wait()
    assert (tmp_path / "pics clean" / "a.jpg").exists()
    client.post("/tools/convert", data={"folder": str(src), "copy_others": "1"})
    app.tools.wait()
    assert (tmp_path / "pics converted" / "a.jpg").exists()
    assert "Converted 0, copied 1" in client.get("/tools/convert").text


def test_running_job_shows_progress_and_disables_other_tools(web):
    from facegroup.services.tools import Job
    app, client = web
    app.tools.job = Job("convert", "Convert HEIC", "running", done=1234, total=5000)
    assert "1,234 of 5,000" in client.get("/tools/job").text
    for url in ("/tools/convert", "/tools/metadata", "/files/duplicates?view=scan"):
        assert " disabled>" in client.get(url).text, url
    r = client.post("/tools/metadata/check", data={"folder": str(Path(__file__).parent)})
    assert r.status_code == 200 and "Convert HEIC is still running." in r.text


def test_job_panel_and_reveal_only_known_paths(web, tmp_path, monkeypatch):
    _, client = web
    assert client.get("/tools/job").headers.get("HX-Refresh") == "true"  # nothing running
    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))
    assert client.post("/files/reveal", data={"path": r"C:\Windows\System32\cmd.exe"}).status_code == 204
    assert calls == []


def test_right_click_handoff_urls(tmp_path):
    from facegroup.entrypoints import launch
    assert launch.page_url("convert", 'C:"') == "/tools/convert?folder=C%3A%5C"  # Explorer's drive-root quoting
    assert launch.page_url("duplicates", None) == "/files/duplicates"
    assert launch.page_url("nonsense", None) == "/photos"
    launch.write_session(tmp_path, 8765, "tok")
    assert launch.read_session(tmp_path)["token"] == "tok"
    launch.clear_session(tmp_path)
    assert launch.read_session(tmp_path) is None


def test_context_menu_commands():
    from facegroup.entrypoints.context_menu import commands
    cmds = commands(Path(r"Q:\Scripts\facegroup\FaceGroup.cmd"))
    key = r"Software\Classes\Directory\Background\shell\FaceGroup\shell\2convert\command"
    assert cmds[key] == r'"Q:\Scripts\facegroup\FaceGroup.cmd" --page convert --folder "%V"'
    assert cmds[r"Software\Classes\Drive\shell\FaceGroup\shell\4open\command"].endswith("--page photos")


def test_login_goes_to_the_requested_page_only_locally(make_app):
    app = make_app()
    client = TestClient(create_app(app, Scheduler(), TOKEN, PORT), base_url=BASE, follow_redirects=False)
    r = client.get(f"/login?t={TOKEN}&next=%2Ftools%2Fconvert%3Ffolder%3DD%3A%5C")
    assert r.headers["location"] == "/tools/convert?folder=D:%5C"
    assert client.get(f"/login?t={TOKEN}&next=//evil.example").headers["location"] == "/photos"


def test_browse_returns_the_chosen_folder(web, monkeypatch):
    _, client = web
    import facegroup.entrypoints.web.tools_routes as routes
    seen = {}

    def fake_pick(initial, title):
        seen.update(initial=initial, title=title)
        return r"D:\Chosen"

    monkeypatch.setattr(routes, "pick_folder", fake_pick)
    r = client.post("/tools/pick-folder", data={"initial": r"D:\Start", "title": "Choose"})
    assert r.json() == {"path": r"D:\Chosen"} and seen == {"initial": r"D:\Start", "title": "Choose"}
    monkeypatch.setattr(routes, "pick_folder", lambda initial, title: None)
    assert client.post("/tools/pick-folder", data={}).json() == {"path": ""}
    assert 'data-pick="Choose the folder with HEIC photos"' in client.get("/tools/convert").text


def test_redirects_stay_on_this_site():
    from facegroup.entrypoints.web.common import safe_next
    assert safe_next("/tools/convert?folder=D%3A%5C") == "/tools/convert?folder=D%3A%5C"
    for bad in ("//evil.example", "/\evil.example", "https://evil.example", "", None):
        assert safe_next(bad, "/home") == "/home"
