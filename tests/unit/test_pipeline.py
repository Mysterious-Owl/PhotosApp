"""Services + SQLite + real decoder/cache, with a fake face model."""

import os
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from facegroup.adapters.instance_lock import AlreadyRunningError
from facegroup.domain.entities import FileInfo, Stage
from facegroup.services.library import LibraryService
from tests.conftest import FakeAnalyzer, count, make_photo, write_sql


def index(app):
    return app.indexing.run_derive(), app.indexing.run_faces()


def scan_and_index(app, folder):
    root = app.library.add_root(folder)
    report = app.library.scan(root)
    derive, faces = index(app)
    return report, derive, faces


def test_end_to_end_scan_derive_faces(make_app, library_dir, unicode_name):
    make_photo(library_dir / "a.jpg", color=(10, 20, 30))
    make_photo(library_dir / unicode_name / "b.jpg", color=(40, 50, 60))
    make_photo(library_dir / "tiny.jpg", size=(60, 60))  # too small: no face
    app = make_app()
    report, derive, faces = scan_and_index(app, library_dir)
    assert (report.seen, report.new) == (3, 3)
    assert (derive.done, derive.failed) == (3, 0)
    assert (faces.done, faces.failed) == (3, 0)
    assert count(app, "SELECT COUNT(*) FROM item") == 3
    assert count(app, "SELECT COUNT(*) FROM face") == 2
    assert count(app, "SELECT COUNT(*) FROM face_embedding WHERE model_id=?", FakeAnalyzer.model_id) == 2
    assert count(app, "SELECT COUNT(*) FROM media WHERE content_hash IS NOT NULL") == 3
    # Thumbnails kept; working images removed once each item is fully processed.
    cache = app.settings.cache_dir
    assert len(list((cache / "thumbs").rglob("*.jpg"))) == 3
    assert list((cache / "work").rglob("*.jpg")) == []


def test_rescan_without_changes_does_no_work(make_app, library_dir):
    make_photo(library_dir / "a.jpg")
    app = make_app()
    scan_and_index(app, library_dir)
    loads = FakeAnalyzer.loads
    report, derive, faces = scan_and_index(app, library_dir)
    assert (report.unchanged, report.new, report.changed) == (1, 0, 0)
    assert derive.done == faces.done == 0
    assert FakeAnalyzer.loads == loads  # model not even loaded when there is nothing to do


def test_changed_file_is_reindexed(make_app, library_dir):
    p = make_photo(library_dir / "a.jpg", color=(10, 10, 10))
    app = make_app()
    scan_and_index(app, library_dir)
    first_hash = count(app, "SELECT COUNT(*) FROM media WHERE content_hash IS NOT NULL")
    make_photo(p, color=(250, 250, 250), size=(500, 300))
    os.utime(p, (1_000_000, 1_000_000))
    report, derive, faces = scan_and_index(app, library_dir)
    assert report.changed == 1 and derive.done == 1 and faces.done == 1
    assert first_hash == 1
    assert count(app, "SELECT COUNT(*) FROM media WHERE file_state='active' AND width=500") == 1


def test_moved_file_keeps_its_faces_and_labels(make_app, library_dir):
    p = make_photo(library_dir / "old" / "a.jpg")
    app = make_app()
    scan_and_index(app, library_dir)
    face_id = count(app, "SELECT id FROM face")
    write_sql(app, "INSERT INTO person(id, name) VALUES(1, 'Mom')")
    write_sql(app, "UPDATE face SET person_id=1, assigned_by='user'")
    (library_dir / "new").mkdir()
    shutil.move(p, library_dir / "new" / "a.jpg")
    report, derive, _ = scan_and_index(app, library_dir)
    assert (report.moved, report.new, report.missing) == (1, 0, 0)
    assert derive.done == 0  # nothing re-processed
    assert count(app, "SELECT COUNT(*) FROM media WHERE rel_path='new/a.jpg'") == 1
    assert count(app, "SELECT person_id FROM face WHERE id=?", face_id) == 1


def test_deleted_file_goes_missing_then_is_removed_after_grace(make_app, library_dir):
    p = make_photo(library_dir / "a.jpg")
    app = make_app()
    root = app.library.add_root(library_dir)
    app.library.scan(root)
    index(app)
    p.unlink()
    report = app.library.scan(root)
    assert report.missing == 1 and report.removed == 0
    assert count(app, "SELECT COUNT(*) FROM face") == 1  # still there while missing
    later = LibraryService(app.store, app.library._files, missing_grace_days=30,
                           now=lambda: datetime.now(timezone.utc) + timedelta(days=31))
    assert later.scan(root).removed == 1
    assert count(app, "SELECT COUNT(*) FROM media") == 0
    assert count(app, "SELECT COUNT(*) FROM face") == 0  # cascaded


def test_disconnected_drive_does_not_mark_library_missing(make_app, tmp_path):
    lib = tmp_path / "usb"
    make_photo(lib / "a.jpg")
    app = make_app()
    root = app.library.add_root(lib)
    app.library.scan(root)
    shutil.rmtree(lib)
    report = app.library.scan(root)
    assert report.errors and report.missing == 0
    assert count(app, "SELECT COUNT(*) FROM media WHERE file_state='missing'") == 0


class OnlineOnlySource:
    """Wraps the real file source and marks chosen files as OneDrive online-only."""

    def __init__(self, inner, online: set[str]):
        self.inner, self.online, self.hashed = inner, online, []

    def walk(self, root):
        for f in self.inner.walk(root):
            yield FileInfo(f.rel_path, f.size, f.mtime, f.rel_path in self.online)

    def content_hash(self, path):
        self.hashed.append(path.name)
        return self.inner.content_hash(path)


def test_online_only_files_are_skipped_until_hydrated(make_app, library_dir):
    make_photo(library_dir / "local.jpg")
    make_photo(library_dir / "cloud.jpg", color=(1, 2, 3))
    app = make_app()
    source = OnlineOnlySource(app.library._files, {"cloud.jpg"})
    lib = LibraryService(app.store, source)
    root = lib.add_root(library_dir)
    r = lib.scan(root)
    assert (r.new, r.online_only) == (1, 1)
    derive, _ = index(app)
    assert derive.done == 1
    assert count(app, "SELECT COUNT(*) FROM media WHERE file_state='online_only'") == 1
    source.online.clear()  # user opened the file; OneDrive downloaded it
    r = lib.scan(root)
    assert r.new == 1
    derive, _ = index(app)
    assert derive.done == 1
    assert source.hashed == []  # no file was hashed (and so downloaded) to look for moves


def test_corrupt_file_fails_once_and_permanently(make_app, library_dir):
    make_photo(library_dir / "good.jpg")
    (library_dir / "bad.jpg").write_bytes(b"garbage")
    app = make_app()
    _, derive, _ = scan_and_index(app, library_dir)
    assert (derive.done, derive.failed) == (1, 1)
    assert count(app, "SELECT attempts FROM job WHERE state='dead'") == 1
    assert "bad.jpg" in app.store.dead_jobs()[0][0]


def test_transient_failures_retry_then_die(make_app, library_dir):
    make_photo(library_dir / "a.jpg", color=(7, 7, 7))
    app = make_app(analyzer=lambda: FakeAnalyzer(fail_on_color=(7, 7, 7)))
    scan_and_index(app, library_dir)
    for _ in range(3):
        app.indexing.run_faces()
    assert count(app, "SELECT COUNT(*) FROM job WHERE stage='faces' AND state='dead' AND attempts=3") == 1
    assert app.store.retry_dead() == 1


def test_resume_after_crash(make_app, settings, library_dir):
    for i in range(5):
        make_photo(library_dir / f"{i}.jpg", color=(i * 40, 0, 0))
    app = make_app()
    root = app.library.add_root(library_dir)
    app.library.scan(root)
    # Simulate a crash: two jobs leased (running) and the process dies without finishing them.
    leased = app.store.lease_jobs(Stage.DERIVE, 2, lease_seconds=900, max_attempts=3)
    assert len(leased) == 2
    app.close()

    app2 = make_app()
    assert app2.indexing.recover() == 2
    derive, faces = index(app2)
    assert derive.done == 5 and faces.done == 5


def test_repeated_crashes_on_one_file_stop_after_max_attempts(make_app, library_dir):
    make_photo(library_dir / "killer.jpg")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    app.close()
    for _ in range(3):  # each run leases the job and "crashes"
        a = make_app()
        a.indexing.recover()
        assert len(a.store.lease_jobs(Stage.DERIVE, 1, 900, 3)) == 1
        a.close()
    a = make_app()
    a.indexing.recover()
    assert count(a, "SELECT COUNT(*) FROM job WHERE state='dead'") == 1
    assert a.store.lease_jobs(Stage.DERIVE, 1, 900, 3) == []


def test_expired_lease_counts_as_an_attempt(make_app, library_dir):
    make_photo(library_dir / "a.jpg")
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    for _ in range(3):
        app.store.lease_jobs(Stage.DERIVE, 1, lease_seconds=-1, max_attempts=3)  # lease already expired
    assert app.store.lease_jobs(Stage.DERIVE, 1, 900, 3) == []
    assert count(app, "SELECT COUNT(*) FROM job WHERE state='dead'") == 1


def test_only_one_process_at_a_time(make_app):
    make_app()
    with pytest.raises(AlreadyRunningError):
        make_app()


def test_redetection_keeps_labelled_face_rows(make_app, library_dir):
    make_photo(library_dir / "a.jpg")
    app = make_app()
    scan_and_index(app, library_dir)
    face_id = count(app, "SELECT id FROM face")
    write_sql(app, "INSERT INTO person(id, name) VALUES(1, 'Dad')")
    write_sql(app, "UPDATE face SET person_id=1, assigned_by='user'")
    write_sql(app, "UPDATE job SET state='pending', attempts=0 WHERE stage='faces'")
    _, faces = index(app)  # working image was cleaned up, so this also exercises the rebuild path
    assert faces.done == 1
    assert count(app, "SELECT COUNT(*) FROM face") == 1
    assert count(app, "SELECT id FROM face WHERE person_id=1") == face_id


def test_should_stop_leaves_remaining_jobs_pending(make_app, library_dir):
    for i in range(6):
        make_photo(library_dir / f"{i}.jpg", color=(i * 30, 0, 0))
    app = make_app()
    app.library.scan(app.library.add_root(library_dir))
    calls = {"n": 0}

    def stop_after_first_batch():
        calls["n"] += 1
        return calls["n"] > 1

    app.indexing._should_stop = stop_after_first_batch
    report = app.indexing.run_derive()
    assert 0 < report.done < 6
    assert count(app, "SELECT COUNT(*) FROM job WHERE stage='derive' AND state='pending'") == 6 - report.done
