"""Batch D: timeline, on this day, events, saved searches (synthetic data)."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from facegroup.domain.browse import Event, by_month, events, month_label, on_this_day
from facegroup.entrypoints.scheduler import Scheduler
from facegroup.entrypoints.web.server import create_app
from facegroup.ports.search import SearchFilters
from facegroup.services.browse import BrowseService
from tests.conftest import count, make_photo, write_sql
from tests.unit.test_web import BASE, PORT, TOKEN, login


# -- domain ---------------------------------------------------------------------------------------------------

def test_by_month_and_label():
    dated = [(3, "2024-05-02T10:00:00"), (2, "2024-05-01T09:00:00"), (1, "2023-12-31T23:00:00")]
    assert by_month(dated) == [("2024-05", [3, 2]), ("2023-12", [1])]
    assert month_label("2023-12") == "December 2023"


def test_on_this_day_previous_years_only():
    dated = [(1, "2025-10-25T10:00:00"), (2, "2023-10-25T08:00:00"), (3, "2023-10-24T08:00:00"), (4, "2026-10-25T09:00:00")]
    assert on_this_day(dated, date(2026, 10, 25)) == [(1, [1]), (3, [2])]
    assert on_this_day(dated, date(2026, 10, 25), window_days=1) == [(1, [1]), (3, [2, 3])]
    assert on_this_day([(9, "2024-02-29T12:00:00")], date(2025, 2, 28)) == [(1, [9])]  # leap day


def test_events_split_on_gaps_and_need_enough_photos():
    day1 = [(i, f"2024-07-01T{9 + i:02d}:00:00") for i in range(6)]  # 6 photos within a day
    day3 = [(10 + i, f"2024-07-03T{10 + i:02d}:00:00") for i in range(7)]
    lonely = [(99, "2024-08-01T12:00:00")]
    found = events(day1 + day3 + lonely)
    assert [len(e.media_ids) for e in found] == [7, 6]  # newest first; the lone photo is not an event
    assert found[0].title == "3 July 2024"
    assert Event("2024-07-30T10:00:00", "2024-08-02T10:00:00").title == "30 Jul – 2 Aug 2024"


def test_saved_search_query_is_sanitised():
    assert BrowseService.safe_query("q=beach&kind=photos&evil=1&next=//x") == "q=beach&kind=photos"


# -- service + web ----------------------------------------------------------------------------------------------

@pytest.fixture
def library(make_app, library_dir):
    for i in range(8):
        make_photo(library_dir / "Goa trip" / f"{i}.jpg", color=(20 * i, 100, 150))
    make_photo(library_dir / "Screenshots" / "Screenshot 1.png", color=(240, 240, 240), size=(1920, 1080), fmt="PNG")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.indexing.run_all()
    for i in range(8):  # a trip in July 2023, one photo per hour
        write_sql(app, "UPDATE media SET taken_at=? WHERE rel_path=?", f"2023-07-14T{9 + i:02d}:00:00", f"Goa trip/{i}.jpg")
    app.organize.refresh()
    return app


def test_timeline_hides_clutter_by_default(library):
    tl = library.browse.timeline(SearchFilters(kind="photos"))
    shown = [t.media_id for _, _, tiles in tl["months"] for t in tiles]
    shot = count(library, "SELECT id FROM media WHERE rel_path LIKE 'Screenshots%'")
    assert len(shown) == 8 and shot not in shown
    everything = library.browse.timeline(SearchFilters())
    assert shot in [t.media_id for _, _, tiles in everything["months"] for t in tiles]
    assert tl["months"][0][1] == "July 2023"


def test_events_find_the_trip_with_its_folder(library):
    found = library.browse.events(SearchFilters(kind="photos"))
    assert len(found) == 1 and found[0]["count"] == 8 and found[0]["folder"] == "Goa trip"
    assert found[0]["title"] == "14 July 2023"
    assert library.browse.event(SearchFilters(kind="photos"), found[0]["key"])["tiles"]


def test_memories_on_the_trip_anniversary(library):
    m = library.browse.memories(SearchFilters(kind="photos"), today=date(2026, 7, 14))
    assert m and m[0][0] == 3 and len(m[0][1]) == 8


def test_web_pages_and_saved_searches(library):
    client = TestClient(create_app(library, Scheduler(), TOKEN, PORT), base_url=BASE, follow_redirects=False)
    login(client)
    assert client.get("/").headers["location"] == "/photos"
    page = client.get("/photos").text
    assert "July 2023" in page and "Show screenshots and other clutter too" in page
    assert "Goa trip" in client.get("/events").text
    key = library.browse.events(SearchFilters(kind="photos"))[0]["key"]
    assert client.get("/event", params={"key": key}).status_code == 200
    r = client.post("/search/save", data={"name": "Trip", "q": "", "kind": "photos", "evil": "x", "current": "kind=photos"})
    assert r.status_code == 303
    saved = library.browse.saved()
    assert saved[0]["name"] == "Trip" and saved[0]["query"] == "kind=photos"
    assert client.get(f"/saved/{saved[0]['id']}").headers["location"] == "/search?kind=photos"
    assert "Trip" in client.get("/search?kind=photos").text
    client.post(f"/search/saved/{saved[0]['id']}/delete")
    assert library.browse.saved() == []
    assert client.post("/search/save", data={"name": " "}).status_code == 200  # needs a name: friendly error
