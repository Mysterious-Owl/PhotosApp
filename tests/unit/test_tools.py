"""File tools on synthetic files in temporary folders: duplicate rules, convert, metadata, move and restore."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from facegroup.adapters.tools import images, walk
from facegroup.domain.files import (
    Copy,
    converted_name,
    decide,
    folder_pairs,
    holding_path,
    is_copy_name,
    pair_key,
    sibling_folder,
)
from facegroup.services.tools import (
    ConvertOptions,
    MetadataOptions,
    ToolRunner,
    run_convert,
    run_metadata_check,
    run_metadata_clean,
)
from tests.conftest import make_photo


def C(path, size=100, mtime=1.0):
    return Copy(path, size, mtime)


# -- domain ---------------------------------------------------------------------------------------------------

def test_copy_names():
    assert is_copy_name(r"D:\a\IMG_1 (1).jpg") and is_copy_name(r"D:\a\IMG_1 - Copy.jpg")
    assert is_copy_name(r"D:\a\IMG_1 - Copy (2).jpg") and is_copy_name(r"D:\a\IMG_1 copy.jpg")
    assert not is_copy_name(r"D:\a\IMG_1.jpg") and not is_copy_name(r"D:\a\Copyright.jpg")


def test_same_folder_keeps_the_plain_name():
    d = decide([C(r"D:\a\IMG_1 (1).jpg", mtime=0), C(r"D:\a\IMG_1.jpg", mtime=5)], {})
    assert d.reason == "same_folder" and d.keep.path.endswith("IMG_1.jpg")


def test_folder_preference_applies_to_every_group_and_ignores_case():
    prefs = {pair_key(r"D:\Backup", r"Q:\Photos"): r"q:\photos"}
    d = decide([C(r"D:\backup\x.jpg"), C(r"Q:\PHOTOS\x.jpg")], prefs)
    assert d.reason == "rule" and d.keep.path == r"Q:\PHOTOS\x.jpg"
    # Three folders need the kept folder to win against both others.
    three = [C(r"D:\backup\x.jpg"), C(r"Q:\Photos\x.jpg"), C(r"E:\old\x.jpg")]
    assert decide(three, prefs).reason == "suggested"
    prefs[pair_key(r"E:\old", r"Q:\Photos")] = r"q:\photos"
    assert decide(three, prefs).keep.path == r"Q:\Photos\x.jpg"


def test_choice_beats_preference_and_suggestion_avoids_backup_folders():
    prefs = {pair_key(r"D:\Backup", r"Q:\Photos"): r"q:\photos"}
    copies = [C(r"D:\Backup\x.jpg"), C(r"Q:\Photos\x.jpg")]
    assert decide(copies, prefs, choice=r"D:\Backup\x.jpg").keep.path == r"D:\Backup\x.jpg"
    s = decide([C(r"D:\Backup\x.jpg", mtime=0), C(r"Q:\Trips\x.jpg", mtime=9)], {})
    assert s.reason == "suggested" and not s.confirmed and s.keep.path == r"Q:\Trips\x.jpg"


def test_folder_pairs_rank_by_space_and_show_decisions():
    groups = [[C(r"A:\x\1.jpg", 10), C(r"B:\y\1.jpg", 10)], [C(r"A:\x\2.jpg", 500), C(r"B:\y\2.jpg", 500)],
              [C(r"A:\x\3.jpg", 5), C(r"C:\z\3.jpg", 5)]]
    pairs = folder_pairs(groups, {pair_key(r"A:\x", r"C:\z"): r"a:\x"})
    assert [(p.a, p.b, p.groups, p.extra_bytes) for p in pairs] == [(r"A:\x", r"B:\y", 2, 510), (r"A:\x", r"C:\z", 1, 5)]
    assert pairs[0].winner is None and pairs[1].winner == r"A:\x"


def test_holding_path_stays_on_the_drive_and_never_clashes():
    assert holding_path(r"D:\Trip\a.jpg") == r"D:\FaceGroup removed\Trip\a.jpg"
    taken = {r"d:\facegroup removed\trip\a.jpg"}
    assert holding_path(r"D:\Trip\a.jpg", taken) == r"D:\FaceGroup removed\Trip\a (2).jpg"


def test_output_names():
    assert converted_name("IMG_1.HEIC", "png") == "IMG_1.png" and converted_name("a.mov", "png") == "a.mov"
    assert sibling_folder(r"D:\Trip", "converted") == r"D:\Trip converted"
    assert sibling_folder("D:\\", "converted") == r"D:\FaceGroup converted"


# -- convert and metadata -----------------------------------------------------------------------------------------

def _exif_photo(path: Path, fmt: str | None = None, color=(200, 10, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ex = Image.Exif()
    ex[0x0112] = 6  # rotated: stored landscape, shown portrait
    ex.get_ifd(images.EXIF_IFD)[images.EXIF_DATETIME_ORIGINAL] = "2023:05:01 10:00:00"
    ex.get_ifd(images.GPS_IFD)[1] = "N"
    Image.new("RGB", (40, 30), color).save(path, format=fmt, exif=ex.tobytes())
    return path


def test_convert_heic_keeps_exif_and_never_touches_originals(tmp_path):
    src = tmp_path / "trip"
    heic = _exif_photo(src / "a.heic", "HEIF")
    make_photo(src / "b.jpg")
    before = heic.read_bytes()
    out = tmp_path / "trip converted"
    runner = ToolRunner()
    runner.start("convert", "Convert", lambda r: run_convert(ConvertOptions(src, out, "jpg", copy_others=True), r))
    job = runner.wait()
    assert job.state == "done" and job.result["counts"] == {"converted": 1, "copied": 1}
    assert sorted(p.name for p in out.iterdir()) == ["a.jpg", "b.jpg"]
    assert images.inspect(out / "a.jpg").date_taken and images.inspect(out / "a.jpg").gps
    assert heic.read_bytes() == before
    # A second run skips what is already there.
    runner.start("convert", "Convert", lambda r: run_convert(ConvertOptions(src, out, "jpg", copy_others=True), r))
    assert runner.wait().result["counts"] == {"skipped": 2}
    assert not list(out.glob("*.part"))


def test_metadata_check_and_clean(tmp_path):
    src = tmp_path / "pics"
    _exif_photo(src / "a.jpg")
    _exif_photo(src / "sub" / "b.png")
    make_photo(src / "c.webp", fmt="WEBP")
    runner = ToolRunner()
    runner.start("check", "Check", lambda r: run_metadata_check(src, True, r))
    res = runner.wait().result
    assert res["files"] == 3 and res["tally"]["gps"] == 2 and res["with_gps"] == ["a.jpg", "sub\\b.png" if os.name == "nt" else "sub/b.png"]
    out = tmp_path / "pics clean"
    runner.start("clean", "Clean", lambda r: run_metadata_clean(MetadataOptions(src, out, keep_date=True), r))
    assert runner.wait().result["counts"] == {"cleaned": 3}
    for p in (out / "a.jpg", out / "sub" / "b.png"):
        r = images.inspect(p)
        assert not r.gps and r.date_taken and r.exif_tags <= 2
        assert Image.open(p).size == (30, 40)  # turned upright before the orientation tag went
    images.strip_metadata(src / "a.jpg", tmp_path / "bare.jpg")
    assert images.inspect(tmp_path / "bare.jpg").clean


# -- duplicates: scan, decide, move, restore ----------------------------------------------------------------------

@pytest.fixture
def dupe_tree(tmp_path):
    root = tmp_path / "drive"
    a, b = root / "Photos", root / "Backup"
    make_photo(a / "x.jpg", color=(1, 2, 3))
    shutil.copy2(a / "x.jpg", b / "x.jpg") if b.mkdir(parents=True) is None else None
    make_photo(a / "y.jpg", color=(9, 9, 9))
    shutil.copy2(a / "y.jpg", b / "y.jpg")
    shutil.copy2(a / "y.jpg", a / "y (1).jpg")  # same-folder copy
    make_photo(a / "unique.jpg", color=(50, 60, 70))
    (root / "empty" / "nested").mkdir(parents=True)
    return root


def test_duplicates_end_to_end(make_app, dupe_tree, monkeypatch):
    # Holding folders go under the drive root; point them at the temp tree instead.
    from facegroup.services import tools
    monkeypatch.setattr(tools, "holding_path",
                        lambda p, taken=None: str(dupe_tree / "FaceGroup removed" / Path(p).relative_to(dupe_tree)))
    app = make_app()
    dup, runner = app.duplicates, app.tools
    runner.start("dupes", "Scan", lambda r: dup.run_scan([dupe_tree], media_only=True, min_size=1, runner=r))
    assert runner.wait().state == "done"
    s = dup.summary()
    assert s["groups"] == 2 and s["scan"]["files"] == 6
    assert s["by_reason"] == {"suggested": 2}  # y has a same-folder copy but spans two folders
    assert dup.empty_folders() == [str(dupe_tree / "empty"), str(dupe_tree / "empty" / "nested")]

    pair = dup.pairs()[0]
    dup.prefer(pair.a, pair.b, keep=str(dupe_tree / "Photos"))
    s = dup.summary()
    assert s["ready"] == 2 and s["ready_files"] == 3 and s["suggested"] == 0

    runner.start("remove", "Remove", lambda r: dup.run_remove(False, r))
    assert runner.wait().state == "done"
    left = sorted(p.relative_to(dupe_tree).as_posix() for p in dupe_tree.rglob("*.jpg"))
    assert [p for p in left if not p.startswith("FaceGroup removed")] == ["Photos/unique.jpg", "Photos/x.jpg", "Photos/y.jpg"]
    assert dup.summary()["groups"] == 0

    batch = dup.batches()[0]
    assert batch["files"] == 3 and batch["still_removed"] == 3
    runner.start("restore", "Restore", lambda r: dup.run_restore(batch["id"], r))
    assert runner.wait().state == "done"
    assert (dupe_tree / "Backup" / "x.jpg").exists() and (dupe_tree / "Photos" / "y (1).jpg").exists()
    assert not (dupe_tree / "FaceGroup removed").exists()  # emptied folders are tidied away
    assert dup.batches()[0]["still_removed"] == 0


def test_remove_skips_files_changed_since_the_scan(make_app, dupe_tree, monkeypatch):
    from facegroup.services import tools
    monkeypatch.setattr(tools, "holding_path",
                        lambda p, taken=None: str(dupe_tree / "FaceGroup removed" / Path(p).relative_to(dupe_tree)))
    app = make_app()
    dup, runner = app.duplicates, app.tools
    runner.start("dupes", "Scan", lambda r: dup.run_scan([dupe_tree], True, 1, r))
    runner.wait()
    (dupe_tree / "Backup" / "x.jpg").write_bytes(b"edited since")
    runner.start("remove", "Remove", lambda r: dup.run_remove(True, r))
    job = runner.wait()
    assert (dupe_tree / "Backup" / "x.jpg").read_bytes() == b"edited since"
    assert any("changed since the scan" in s or "copy to keep" in s for s in job.result["skipped"])


def test_empty_folders_removed_and_restored(make_app, dupe_tree):
    app = make_app()
    dup, runner = app.duplicates, app.tools
    runner.start("dupes", "Scan", lambda r: dup.run_scan([dupe_tree], True, 1, r))
    runner.wait()
    assert dup.remove_empty_folders(dup.empty_folders()) == 2
    assert not (dupe_tree / "empty").exists()
    runner.start("restore", "Restore", lambda r: dup.run_restore(dup.batches()[0]["id"], r))
    runner.wait()
    assert (dupe_tree / "empty" / "nested").is_dir()


def test_walk_skips_the_holding_folder(tmp_path):
    make_photo(tmp_path / "FaceGroup removed" / "a.jpg")
    make_photo(tmp_path / "b.jpg")
    assert [Path(f.path).name for f in walk.walk([tmp_path], None)] == ["b.jpg"]


def test_runner_refuses_a_second_job():
    import threading
    gate = threading.Event()
    runner = ToolRunner()
    runner.start("a", "A", lambda r: gate.wait(5) and {})
    from facegroup.services.tools import Busy
    with pytest.raises(Busy):
        runner.start("b", "B", lambda r: {})
    gate.set()
    runner.wait()


# -- HEIC + PNG: keep the PNG ---------------------------------------------------------------------------------------

def test_png_is_preferred_over_heic():
    from facegroup.domain.files import converted_candidates
    d = decide([C(r"D:\a\IMG_1.HEIC", 900), C(r"D:\a converted\IMG_1.png", 3000)], {}, converted=True)
    assert d.reason == "converted" and d.keep.path.endswith(".png") and not d.confirmed
    # Byte-identical copies differing only by extension (a renamed file): the PNG name is kept too.
    assert decide([C(r"D:\a\x.heic"), C(r"D:\a\x.png")], {}).keep.path.endswith(".png")
    pairs = converted_candidates([C(r"D:\t\IMG_1.HEIC")], [C(r"E:\other\img_1.png"), C(r"D:\t converted\IMG_1.png")])
    assert [p.path for _, p in pairs] == [r"D:\t converted\IMG_1.png", r"E:\other\img_1.png"]  # nearby first


def test_scan_pairs_heic_with_its_png_but_not_a_namesake(make_app, tmp_path, monkeypatch):
    import facegroup.services.tools as tools
    monkeypatch.setattr(tools, "holding_path",
                        lambda p, taken=None: str(tmp_path / "FaceGroup removed" / Path(p).relative_to(tmp_path)))
    trip = tmp_path / "trip"
    heic = _exif_photo(trip / "IMG_1.heic", "HEIF")
    images.convert_heic(heic, tmp_path / "trip converted" / "IMG_1.png", "png")
    _exif_photo(trip / "IMG_2.heic", "HEIF", color=(10, 10, 200))
    make_photo(tmp_path / "elsewhere" / "IMG_2.png", size=(40, 30), color=(0, 200, 0))  # same name, other picture
    app = make_app()
    dup, runner = app.duplicates, app.tools
    runner.start("dupes", "Scan", lambda r: dup.run_scan([tmp_path], True, 0, r))
    assert runner.wait().state == "done"
    s = dup.summary()
    assert s["converted"] == 1 and s["ready"] == 0
    assert dup.pairs() == []  # decided by format, not by folder
    runner.start("remove", "Remove", lambda r: dup.run_remove(False, r))
    runner.wait()
    assert heic.exists()  # not without asking
    runner.start("remove", "Remove", lambda r: dup.run_remove(False, r, include_converted=True))
    runner.wait()
    assert not heic.exists() and (tmp_path / "trip converted" / "IMG_1.png").exists()
    assert (trip / "IMG_2.heic").exists()
