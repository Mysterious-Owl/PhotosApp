"""Web UI: security rules and the main flows, through HTTP."""

import pytest
from fastapi.testclient import TestClient

from facegroup.entrypoints.scheduler import Scheduler
from facegroup.entrypoints.web.server import COOKIE, create_app
from tests.conftest import count, make_photo, write_sql
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


class TestSearchPages:
    @pytest.fixture
    def indexed(self, make_app, library_dir):
        make_photo(library_dir / "Goa 2023" / "sea.jpg", color=(0, 0, 250))
        make_photo(library_dir / "Goa 2023" / "fire.jpg", color=(250, 0, 0))
        app = make_app()
        app.library.scan(app.library.add_root(library_dir))
        for run in (app.indexing.run_derive, app.indexing.run_faces, app.indexing.run_embed):
            run()
        client = TestClient(create_app(app, Scheduler(), TOKEN, PORT), base_url=BASE, follow_redirects=False)
        login(client)
        return app, client

    def test_search_page_shows_ranked_results(self, indexed):
        app, client = indexed
        html = client.get("/search?q=red").text
        fire = count(app, "SELECT id FROM media WHERE rel_path LIKE '%fire.jpg'")
        sea = count(app, "SELECT id FROM media WHERE rel_path LIKE '%sea.jpg'")
        assert html.index(f"/photos/{fire}") < html.index(f"/photos/{sea}")

    def test_folder_filter_and_empty_state(self, indexed):
        app, client = indexed
        root = app.library.roots()[0].id
        html = client.get(f"/search?q=red&folder={root}:Nowhere").text
        assert "No photos match" in html
        assert "Goa 2023" in client.get("/search").text  # folder offered as a filter

    def test_photo_page_and_similar(self, indexed):
        app, client = indexed
        fire = count(app, "SELECT id FROM media WHERE rel_path LIKE '%fire.jpg'")
        page = client.get(f"/photos/{fire}")
        assert page.status_code == 200 and "Find similar photos" in page.text and "fire.jpg" in page.text
        sim = client.get(f"/search?similar={fire}").text
        assert "Photos like" in sim and f"/photos/{fire}\"" not in sim.split("Photos like")[1].split("</h1>")[1]

    def test_search_input_is_escaped(self, indexed):
        _, client = indexed
        html = client.get('/search?q=<script>alert(1)</script>').text
        assert "<script>alert(1)</script>" not in html


class TestCombineLabelsAndPhotos:
    def test_unnamed_people_get_face_numbers(self, env):
        app, _, client = env
        login(client)
        pid = app.people.accept(app.people.suggestions()[0].id, "")
        assert f"Face {pid}" in client.get("/people").text
        assert f"Face {pid}" in client.get(f"/people/{pid}").text

    def test_combine_from_the_overview(self, env):
        app, _, client = env
        login(client)
        s = app.people.suggestions()
        named = app.people.accept(s[0].id, "Mom")
        before = app.people.person(named).faces
        absorbed = app.people.suggestion_faces(s[1].id) + app.people.suggestion_faces(s[2].id)
        r = client.post("/people/combine", data={"person": [str(named)], "suggestion": [str(s[1].id), str(s[2].id)]})
        assert r.status_code == 303 and r.headers["location"] == f"/people/{named}"
        assert app.people.person(named).faces == before + len(absorbed)
        assert all(count(app, "SELECT person_id FROM face WHERE id=?", f) == named for f in absorbed)
        client.post("/undo", data={"next": "/people"})
        assert app.people.person(named).faces == before  # one undo reverses the whole combine

    def test_combine_needs_two_things(self, env):
        app, _, client = env
        login(client)
        sid = app.people.suggestions()[0].id
        r = client.post("/people/combine", data={"suggestion": [str(sid)]})
        assert r.status_code == 200 and "at least two" in r.text

    def test_overview_offers_combine_checkboxes(self, env):
        app, _, client = env
        login(client)
        app.people.accept(app.people.suggestions()[0].id, "A")
        html = client.get("/people").text
        assert 'name="suggestion"' in html and 'name="person"' in html and 'form="combine"' in html


def test_faces_link_to_their_photo_with_the_face_outlined(make_app, library_dir):
    make_photo(library_dir / "a.jpg")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.indexing.run_all()
    client = TestClient(create_app(app, Scheduler(), TOKEN, PORT), base_url=BASE, follow_redirects=False)
    login(client)
    face = count(app, "SELECT id FROM face")
    media = count(app, "SELECT id FROM media")
    pid = app.people.split([face], "")
    person_page = client.get(f"/people/{pid}").text
    assert f"/photos/{media}?face={face}" in person_page
    photo = client.get(f"/photos/{media}?face={face}").text
    assert f'id="face{face}"' in photo and "box highlight" in photo
    assert f"Face {pid}" in photo


class TestBreakingUpGroups:
    def test_act_on_picked_faces_only_and_keep_the_rest(self, env):
        app, _, client = env
        login(client)
        sid = app.people.suggestions()[0].id
        faces = app.people.suggestion_faces(sid)
        picked = faces[:2]
        r = client.post(f"/suggestions/{sid}/faces", data={"face": [str(f) for f in picked], "action": "ignore"})
        assert r.status_code == 303 and r.headers["location"] == f"/suggestions/{sid}"  # back to the rest
        hidden = app.people.hidden()
        assert hidden is not None
        assert {count(app, "SELECT person_id FROM face WHERE id=?", f) for f in picked} == {hidden.id}
        rest = app.people.suggestion_faces(sid)
        assert rest and not set(rest) & set(picked)  # the other faces were NOT marked
        r = client.post(f"/suggestions/{sid}/faces", data={"face": [str(f) for f in rest], "action": "new", "name": "Ann"})
        assert r.headers["location"] == "/people"  # group used up
        assert app.people.person(count(app, "SELECT person_id FROM face WHERE id=?", rest[0])).name == "Ann"

    def test_faces_from_another_group_are_ignored(self, env):
        app, _, client = env
        login(client)
        s = app.people.suggestions()
        other = app.people.suggestion_faces(s[1].id)[0]
        client.post(f"/suggestions/{s[0].id}/faces", data={"face": [str(other)], "action": "new"})
        assert count(app, "SELECT person_id FROM face WHERE id=?", other) is None

    def test_overview_has_no_whole_group_not_a_person_and_links_the_hidden_group(self, env):
        app, _, client = env
        login(client)
        assert "/ignore" not in client.get("/people").text
        app.people.not_a_person(face_ids=app.people.suggestion_faces(app.people.suggestions()[0].id)[:1])
        html = client.get("/people").text
        assert f"/people/{app.people.hidden().id}" in html and "Not a person: 1" in html

    def test_hidden_page_can_send_faces_back(self, env):
        app, _, client = env
        login(client)
        face = app.people.suggestion_faces(app.people.suggestions()[0].id)[0]
        app.people.not_a_person(face_ids=[face])
        hid = app.people.hidden().id
        page = client.get(f"/people/{hid}").text
        assert "This is a person" in page and "Forget this person" not in page
        client.post(f"/people/{hid}/faces", data={"face": [str(face)], "action": "restore"})
        assert count(app, "SELECT person_id FROM face WHERE id=?", face) != hid
        app.people.refresh()
        assert count(app, "SELECT person_id FROM face WHERE id=?", face) != hid  # not pulled back automatically

    def test_review_page_shows_parts_for_a_mixed_group(self, env):
        app, lib, client = env
        login(client)
        a, b = app.people.suggestions()[:2]
        faces_a, faces_b = app.people.suggestion_faces(a.id), app.people.suggestion_faces(b.id)
        # Force a mixed group: put one of B's faces into A's suggestion.
        write_sql(app, "INSERT INTO suggestion_face(suggestion_id, face_id) VALUES(?, ?)", a.id, faces_b[0])
        html = client.get(f"/suggestions/{a.id}").text
        assert "Part A" in html and "Part B" in html and "Select this part" in html
        part_b = html.split("Part B")[1]
        assert f'value="{faces_b[0]}"' in part_b and not any(f'value="{f}"' in part_b for f in faces_a)


class TestQuickCheck:
    def test_check_page_and_answers(self, env):
        app, lib, client = env
        login(client)
        faces = lib.faces_of(0)
        a = app.people.split(faces[:4], "A")
        b = app.people.split(faces[4:], "")
        html = client.get("/check").text
        assert "What FaceGroup has learned" in html
        assert "the same person?" in html  # A and Face b look alike
        r = client.post("/check/answer", data={"kind": "same_person", "person": str(a), "other": str(b), "answer": "no"})
        assert r.status_code == 303 and r.headers["location"] == "/check"
        assert "the same person?" not in client.get("/check").text  # not asked again

    def test_answers_validate_input(self, env):
        _, _, client = env
        login(client)
        for data in ({"kind": "is_person", "face": "x", "person": "1"}, {"kind": "nope"}, {}):
            assert client.post("/check/answer", data=data).status_code == 303  # ignored, no crash

    def test_nav_has_check(self, env):
        _, _, client = env
        login(client)
        assert 'href="/check"' in client.get("/people").text


class TestTidy:
    @pytest.fixture
    def tidy(self, make_app, library_dir):
        from tests.unit.test_organize import organize_lib
        app = organize_lib(make_app, library_dir)
        client = TestClient(create_app(app, Scheduler(), TOKEN, PORT), base_url=BASE, follow_redirects=False)
        login(client)
        return app, client

    def test_tidy_page_and_keep_a_copy(self, tidy):
        app, client = tidy
        html = client.get("/tidy").text
        assert "Duplicates" in html and "Screenshots" in html and "1 set" in html
        members = app.organize.duplicate_sets()[0]["members"]
        loser = [m for m in members if not m["keeper"]][0]["id"]
        client.post("/duplicates/keep", data={"keeper": str(loser)})  # user prefers another copy
        assert count(app, "SELECT hidden FROM media WHERE id=?", loser) == 0
        assert count(app, "SELECT COUNT(*) FROM media WHERE hidden=1") == len(members) - 1

    def test_photo_flags_and_category(self, tidy):
        app, client = tidy
        m = count(app, "SELECT id FROM media WHERE rel_path LIKE '%b1.jpg'")
        client.post(f"/photos/{m}/flags", data={"favorite": "1"})
        client.post(f"/photos/{m}/flags", data={"rating": "4"})
        client.post(f"/photos/{m}/flags", data={"category": "meme"})
        f = app.organize_store.flags(m)
        assert f["favorite"] == 1 and f["rating"] == 4 and f["category"] == "meme" and f["source"] == "user"
        client.post(f"/photos/{m}/flags", data={"rating": "9"})  # invalid -> cleared, never stored
        assert app.organize_store.flags(m)["rating"] is None
        page = client.get(f"/photos/{m}").text
        assert "Burst: 2 similar shots" in page and "★ Favourite" in page

    def test_search_kind_filter_and_burst_badge(self, tidy):
        app, client = tidy
        html = client.get("/search?kind=screenshot").text
        shot = count(app, "SELECT id FROM media WHERE rel_path LIKE '%Screenshot 2024.png'")
        assert f"/photos/{shot}" in html
        assert 'class="stackn"' in client.get("/search?q=red").text
