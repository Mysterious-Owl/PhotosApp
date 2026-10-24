"""Web UI: security rules and the main flows, through HTTP."""

import pytest
from fastapi.testclient import TestClient

from facegroup.entrypoints.scheduler import Scheduler
from facegroup.entrypoints.web.server import COOKIE, create_app
from tests.conftest import count, make_photo
from tests.unit.test_people_service import Library

PORT = 8765
TOKEN = "t" * 43
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture
def env(make_app):
    app = make_app()
    lib = Library(app)
    app.people.refresh()
    scheduler = Scheduler()
    scheduler.attach(app)
    web = create_app(app, scheduler, TOKEN, PORT)
    client = TestClient(web, base_url=BASE, follow_redirects=False)
    return app, lib, client


def login(client):
    r = client.get(f"/login?t={TOKEN}")
    assert r.status_code == 303
    return client


class TestSecurity:
    def test_locked_without_session(self, env):
        _, _, client = env
        assert client.get("/people").status_code == 401
        assert client.get("/faces/1.jpg").status_code == 401
        assert client.post("/undo").status_code == 401

    def test_wrong_token_does_not_unlock(self, env):
        _, _, client = env
        assert client.get("/login?t=nope").status_code == 401
        assert COOKIE not in client.cookies

    def test_login_sets_strict_httponly_cookie(self, env):
        _, _, client = env
        r = client.get(f"/login?t={TOKEN}")
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert client.get("/people").status_code == 200

    def test_dns_rebinding_host_is_refused(self, env):
        _, _, client = env
        login(client)
        r = client.get("/people", headers={"host": "evil.example:8765"})
        assert r.status_code == 400
        assert client.get("/people", headers={"host": "127.0.0.1:9999"}).status_code == 400  # port matters

    def test_cross_origin_post_is_refused(self, env):
        _, _, client = env
        login(client)
        r = client.post("/undo", headers={"origin": "http://evil.example"})
        assert r.status_code == 403
        assert client.post("/undo", headers={"origin": BASE}).status_code == 303

    def test_redirect_target_cannot_leave_the_site(self, env):
        _, _, client = env
        login(client)
        r = client.post("/undo", data={"next": "//evil.example/x"})
        assert r.headers["location"] == "/people"


class TestFlows:
    def test_overview_lists_suggestions(self, env):
        app, _, client = env
        login(client)
        html = client.get("/people").text
        assert "Who is this?" in html
        assert html.count('class="card suggestion"') == len(app.people.suggestions())

    def test_quick_save_names_whole_group(self, env):
        app, lib, client = env
        login(client)
        s = app.people.suggestions()[0]
        faces = app.people.suggestion_faces(s.id)
        r = client.post(f"/suggestions/{s.id}/accept", data={"all": "1", "name": "Mom", "next": "/people"})
        assert r.status_code == 303
        pid = count(app, "SELECT person_id FROM face WHERE id=?", faces[0])
        assert app.people.person(pid).name == "Mom"
        assert all(count(app, "SELECT person_id FROM face WHERE id=?", f) == pid for f in faces)

    def test_review_unticked_faces_stay_unsorted(self, env):
        app, _, client = env
        login(client)
        s = app.people.suggestions()[0]
        faces = app.people.suggestion_faces(s.id)
        client.post(f"/suggestions/{s.id}/accept", data={"face": [str(f) for f in faces[1:]], "name": "Dad"})
        assert count(app, "SELECT assigned_by='user' FROM face WHERE id=?", faces[0]) in (0, None)

    def test_person_page_actions_and_undo(self, env):
        app, _, client = env
        login(client)
        s = app.people.suggestions()
        a = app.people.accept(s[0].id, "A")
        b = app.people.accept(s[1].id, "B")
        page = client.get(f"/people/{a}")
        assert page.status_code == 200 and "Not them" in page.text
        face = app.people.person_faces(a)[0].id
        client.post(f"/people/{a}/faces", data={"face": [str(face)], "action": "move", "target": str(b)})
        assert count(app, "SELECT person_id FROM face WHERE id=?", face) == b
        client.post("/undo", data={"next": f"/people/{a}"})
        assert count(app, "SELECT person_id FROM face WHERE id=?", face) == a
        client.post(f"/people/{a}/rename", data={"name": "Alice"})
        assert app.people.person(a).name == "Alice"
        client.post(f"/people/{b}/merge", data={"into": str(a)})
        assert app.people.person(b) is None

    def test_forget_needs_typed_confirmation(self, env):
        app, _, client = env
        login(client)
        pid = app.people.accept(app.people.suggestions()[0].id, "X")
        client.post(f"/people/{pid}/forget", data={"confirm": "yes"})
        assert app.people.person(pid) is not None
        client.post(f"/people/{pid}/forget", data={"confirm": "forget"})
        assert app.people.person(pid) is None

    def test_add_folder_rejects_non_folders(self, env, tmp_path):
        _, _, client = env
        login(client)
        r = client.post("/roots", data={"path": str(tmp_path / "does-not-exist")})
        assert r.status_code == 200 and "Not a folder" in r.text

    def test_images_served_by_id(self, make_app, library_dir):
        make_photo(library_dir / "a.jpg")
        app = make_app()
        app.library.scan(app.library.add_root(library_dir))
        app.indexing.run_derive()
        app.indexing.run_faces()
        scheduler = Scheduler()
        client = TestClient(create_app(app, scheduler, TOKEN, PORT), base_url=BASE, follow_redirects=False)
        login(client)
        face = count(app, "SELECT id FROM face")
        media = count(app, "SELECT id FROM media")
        for url in (f"/faces/{face}.jpg", f"/media/{media}/thumb.jpg", f"/media/{media}/photo.jpg"):
            r = client.get(url)
            assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
        assert client.get("/faces/999999.jpg").status_code == 404
        assert client.get("/media/..%2F..%2Fsecret/photo.jpg").status_code in (404, 422)
