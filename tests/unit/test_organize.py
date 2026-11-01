"""Batch A: categories (declutter), duplicates, stacks, per-photo flags. Synthetic images only."""

import os
import shutil

import numpy as np
import pytest
from PIL import Image

from facegroup.adapters.media.phash import dhash, from_signed, to_signed
from facegroup.domain.organize import (Copy, Shot, classify, connected_sets, hash_pairs, pick_keeper,
                                       screenshot_hint, stacks)
from facegroup.ports.search import SearchFilters
from tests.conftest import count, make_photo


# -- domain ---------------------------------------------------------------------------------------------------

def test_hash_pairs_finds_close_hashes_only():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, 200, dtype=np.uint64)
    near = base[:20] ^ np.uint64(0b101)  # 2 bits flipped
    far = base[20:40] ^ np.uint64((1 << 12) - 1)  # 12 bits flipped
    h = np.concatenate([base, near, far])
    pairs = set(hash_pairs(h, 6))
    assert all((i, 200 + i) in pairs for i in range(20))
    assert not any((20 + i, 220 + i) in pairs for i in range(20))


def test_flat_images_are_not_near_duplicates_of_each_other():
    flat = np.array([0, 0, 1, (1 << 64) - 1], dtype=np.uint64)  # blank/black images: degenerate hashes
    assert hash_pairs(flat, 6) == []


def test_hash_pairs_matches_brute_force():
    rng = np.random.default_rng(1)
    h = rng.integers(0, 2**63, 60, dtype=np.uint64)
    h[30:] = h[:30] ^ rng.integers(0, 2**8, 30, dtype=np.uint64)  # up to 8 bits apart
    ok = lambda x: 8 <= bin(int(x)).count("1") <= 56
    brute = {(i, j) for i in range(60) for j in range(i + 1, 60)
             if bin(int(h[i]) ^ int(h[j])).count("1") <= 6 and ok(h[i]) and ok(h[j])}
    assert set(hash_pairs(h, 6)) == brute


def test_connected_sets_and_keeper():
    assert connected_sets(5, [(0, 1), (1, 2)]) == [[0, 1, 2]]
    copies = [Copy(1, 100, False, 50, "a/b/c.jpg"), Copy(2, 400, False, 10, "x.jpg"), Copy(3, 400, True, 5, "y.jpg")]
    assert pick_keeper(copies) == 3  # most pixels, then has a capture date


def test_stacks_need_same_folder_short_gap_and_similar():
    v = np.array([[1, 0], [0.99, 0.141], [1, 0], [0, 1]], np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    shots = [Shot(0, "f", 0), Shot(1, "f", 5), Shot(2, "g", 6), Shot(3, "f", 8)]
    assert stacks(shots, v) == [[0, 1]]  # 2: other folder; 3: different picture


def test_screenshot_hint():
    assert screenshot_hint("Pictures/Screenshots/Screenshot (12).png", 1920, 1080, None)
    assert screenshot_hint("x/IMG_1.PNG", 1170, 2532, None)  # phone screen size, no camera date
    assert not screenshot_hint("x/IMG_1.jpg", 4032, 3024, "2023-01-01T10:00:00")
    assert not screenshot_hint("x/photo.png", 1170, 2532, "2023-01-01T10:00:00")  # has a camera date
    assert screenshot_hint("x/Capture.PNG", 800, 600, None)
    assert not screenshot_hint("Captured moments/IMG_1.jpg", 4032, 3024, None)  # folder names don't count
    assert not screenshot_hint("x/Snippets_beach.jpg", 4032, 3024, None)  # nor partial words


def test_classify_only_calls_clutter_with_a_clear_lead():
    protos = {"photo": np.array([1.0, 0.0]), "screenshot": np.array([0.0, 1.0])}
    imgs = np.array([[0.6, 0.8], [0.7, 0.714]], np.float32)
    res = classify(imgs, protos, margin=0.04)
    assert [r.category for r in res] == ["screenshot", "photo"]  # second lead 0.014 < margin


def test_dhash_survives_resize_and_signed_round_trip():
    rng = np.random.default_rng(2)
    img = (rng.random((300, 400, 3)) * 255).astype(np.uint8)
    img[:, :200] //= 3  # some structure
    small = np.asarray(Image.fromarray(img).resize((200, 150)))
    a, b = dhash(img), dhash(small)
    assert bin(a ^ b).count("1") <= 6
    assert from_signed(to_signed(a)) == a and -(2**63) <= to_signed(a) < 2**63


# -- service + storage ----------------------------------------------------------------------------------------

def textured_photo(path, seed, size=(800, 600)):
    """A photo-like synthetic image with structure (flat colours have degenerate hashes)."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(np.sin(xx / rng.uniform(20, 60) + rng.uniform(0, 6)) * 120 + 128),
                    (np.cos(yy / rng.uniform(20, 60) + rng.uniform(0, 6)) * 120 + 128),
                    (np.sin((xx + yy) / rng.uniform(30, 90)) * 120 + 128)], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=92)
    return path


def organize_lib(make_app, library_dir):
    textured_photo(library_dir / "trip" / "a.jpg", seed=1)
    shutil.copy(library_dir / "trip" / "a.jpg", library_dir / "backup_a.jpg")  # exact copy
    Image.open(library_dir / "trip" / "a.jpg").resize((400, 300)).save(library_dir / "small_a.jpg", quality=70)  # near copy
    make_photo(library_dir / "trip" / "b1.jpg", color=(200, 30, 30), size=(800, 600))
    make_photo(library_dir / "trip" / "b2.jpg", color=(205, 32, 30), size=(640, 480))  # burst of b1
    os.utime(library_dir / "trip" / "b2.jpg", (os.path.getmtime(library_dir / "trip" / "b1.jpg") + 3,) * 2)
    make_photo(library_dir / "Screenshots" / "Screenshot 2024.png", color=(240, 240, 240), size=(1920, 1080), fmt="PNG")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.indexing.run_all()
    app.organize.refresh()
    return app


def mid(app, name):
    return count(app, "SELECT id FROM media WHERE rel_path LIKE ?", f"%{name}")


def test_duplicates_exact_and_near_with_keeper(make_app, library_dir):
    app = organize_lib(make_app, library_dir)
    sets = app.organize.duplicate_sets()
    assert len(sets) == 1
    members = {m["id"] for m in sets[0]["members"]}
    assert members == {mid(app, "trip/a.jpg"), mid(app, "backup_a.jpg"), mid(app, "small_a.jpg")}
    keeper = [m for m in sets[0]["members"] if m["keeper"]][0]
    assert keeper["width"] == 800  # the full-size copy
    app.organize.hide_extra_copies(list(members), keeper["id"])
    assert count(app, "SELECT COUNT(*) FROM media WHERE hidden=1") == 2
    assert count(app, "SELECT COUNT(*) FROM media WHERE hidden=0 AND id=?", keeper["id"]) == 1


def test_bursts_fold_into_one_search_result(make_app, library_dir):
    app = organize_lib(make_app, library_dir)
    stack = app.organize_store.stack_of(mid(app, "b1.jpg"))
    assert {m for m, _ in stack} == {mid(app, "b1.jpg"), mid(app, "b2.jpg")}
    hits = app.search.search("red").hits
    in_stack = [h for h in hits if h.media_id in {mid(app, "b1.jpg"), mid(app, "b2.jpg")}]
    assert len(in_stack) == 1 and in_stack[0].stack_size == 2
    app.organize_store.set_stack_top(mid(app, "b2.jpg"))
    app.organize.refresh()  # the user's pick survives recomputation
    assert dict(app.organize_store.stack_of(mid(app, "b1.jpg")))[mid(app, "b2.jpg")] is True


def test_screenshots_categorised_and_hidden_with_declutter(make_app, library_dir):
    app = organize_lib(make_app, library_dir)
    shot = mid(app, "Screenshot 2024.png")
    assert app.organize_store.flags(shot)["category"] == "screenshot"
    everything = {h.media_id for h in app.search.search("").hits}
    photos_only = {h.media_id for h in app.search.search("", SearchFilters(kind="photos")).hits}
    assert shot in everything and shot not in photos_only
    assert {h.media_id for h in app.search.search("", SearchFilters(kind="screenshot")).hits} == {shot}


def test_user_category_is_never_overwritten(make_app, library_dir):
    app = organize_lib(make_app, library_dir)
    shot = mid(app, "Screenshot 2024.png")
    app.organize.set_category(shot, "photo")
    app.organize.refresh()
    assert app.organize_store.flags(shot)["category"] == "photo"
    with pytest.raises(ValueError):
        app.organize.set_category(shot, "nonsense")


def test_hidden_and_favourite_filters(make_app, library_dir):
    app = organize_lib(make_app, library_dir)
    a = mid(app, "trip/a.jpg")
    app.organize_store.set_flags([a], favorite=True, rating=5)
    assert [h.media_id for h in app.search.search("", SearchFilters(favorites=True)).hits] == [a]
    assert [h.media_id for h in app.search.search("", SearchFilters(min_rating=4)).hits] == [a]
    app.organize_store.set_flags([a], hidden=True)
    assert a not in {h.media_id for h in app.search.search("").hits}
    assert a in {h.media_id for h in app.search.search("", SearchFilters(show_hidden=True)).hits}


def test_photos_indexed_before_hashes_get_backfilled(make_app, library_dir):
    app = organize_lib(make_app, library_dir)
    from tests.conftest import write_sql
    write_sql(app, "UPDATE media SET phash=NULL")
    app.organize.refresh()
    assert count(app, "SELECT COUNT(*) FROM media WHERE phash IS NULL") == 0
