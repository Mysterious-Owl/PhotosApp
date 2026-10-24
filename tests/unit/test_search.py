"""Search: embed stage, filter-first retrieval, keyword matching, fusion, similar photos (fake embedder)."""

import numpy as np
import pytest

from facegroup.domain.search import reciprocal_rank_fusion, top_media
from facegroup.ports.search import SearchFilters
from tests.conftest import FakeEmbedder, count, make_photo, write_sql


def test_rrf_prefers_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [m for m, _ in fused][:2] == [1, 3]
    assert {m for m, _ in fused} == {1, 2, 3, 4}


def test_top_media_keeps_best_frame_per_video():
    media = np.array([10, 10, 20, 30])
    scores = np.array([0.2, 0.9, 0.5, 0.1])
    assert top_media(media, scores, 2) == [(10, 0.9), (20, 0.5)]


@pytest.fixture
def library(make_app, library_dir):
    make_photo(library_dir / "2023 Goa trip" / "beach.jpg", color=(0, 0, 250))
    make_photo(library_dir / "2023 Goa trip" / "sunset.jpg", color=(250, 10, 0))
    make_photo(library_dir / "garden" / "leaves.jpg", color=(10, 240, 10))
    make_photo(library_dir / "garden" / "rose.jpg", color=(240, 30, 30))
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.indexing.run_derive()
    app.indexing.run_faces()
    report = app.indexing.run_embed()
    assert (report.done, report.failed) == (4, 0)
    return app


def names(app, result):
    out = []
    for h in result.hits:
        out.append(app.media.media_label(h.media_id).replace("\\", "/").split("/")[-1])
    return out


def test_embed_stage_stores_one_vector_per_photo_and_cleans_up(library):
    app = library
    assert count(app, "SELECT COUNT(*) FROM item_embedding WHERE model_id=?", FakeEmbedder.model_id) == 4
    assert list((app.settings.cache_dir / "work").rglob("*.jpg")) == []  # all stages done -> cleaned


def test_text_search_ranks_by_meaning(library):
    r = library.search.search("red")
    assert names(library, r)[:2] in (["sunset.jpg", "rose.jpg"], ["rose.jpg", "sunset.jpg"])
    assert r.used_vectors and r.candidates == 4


def test_keyword_search_matches_folder_and_file_names(library):
    r = library.search.search("goa")
    top = names(library, r)
    # "goa" is not a colour (the fake model sees grey), so the folder-name match must lift both Goa photos.
    assert set(top[:2]) == {"beach.jpg", "sunset.jpg"}
    assert all(h.keyword_rank for h in r.hits[:2])


def test_filters_apply_before_ranking(library):
    app = library
    root = app.library.roots()[0]
    r = app.search.search("red", SearchFilters(root_id=root.id, folder="garden"))
    assert names(app, r) == ["rose.jpg", "leaves.jpg"]  # sunset (red) excluded by the folder filter
    assert r.candidates == 2


def test_person_filter(library):
    app = library
    face = count(app, "SELECT f.id FROM face f JOIN item i ON i.id=f.item_id JOIN media m ON m.id=i.media_id "
                      "WHERE m.rel_path LIKE '%rose.jpg'")
    pid = app.people.split([face], "Ann")
    r = app.search.search("", SearchFilters(person_ids=(pid,)))
    assert names(app, r) == ["rose.jpg"]


def test_date_filter_uses_capture_date(library):
    app = library
    write_sql(app, "UPDATE media SET taken_at='2019-05-01T10:00:00' WHERE rel_path LIKE '%leaves.jpg'")
    r = app.search.search("", SearchFilters(date_from="2019-01-01", date_to="2019-12-31"))
    assert names(app, r) == ["leaves.jpg"]


def test_similar_photos_excludes_itself(library):
    app = library
    rose = count(app, "SELECT id FROM media WHERE rel_path LIKE '%rose.jpg'")
    r = app.search.similar(rose)
    assert names(app, r)[0] == "sunset.jpg" and "rose.jpg" not in names(app, r)


def test_keyword_text_is_never_interpreted_as_query_syntax(library):
    for q in ['"', "AND OR NOT", "goa*", "NEAR(a b)", "col:val", "-", "'; DROP TABLE media; --"]:
        library.search.search(q)  # must not raise
    assert count(library, "SELECT COUNT(*) FROM media") == 4


def test_search_reports_photos_not_yet_embedded(make_app, library_dir):
    make_photo(library_dir / "a.jpg")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.indexing.run_derive()  # no embed pass yet
    r = app.search.search("red")
    assert r.missing_embeddings == 1 and not r.used_vectors


def test_photos_indexed_before_search_existed_get_embedded(make_app, library_dir):
    make_photo(library_dir / "a.jpg")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.indexing.run_derive()
    write_sql(app, "DELETE FROM job WHERE stage='embed'")  # as if indexed by an older version
    app.close()
    app = make_app()
    app.indexing.recover()  # notices the missing stage and queues it
    assert app.indexing.run_embed().done == 1
