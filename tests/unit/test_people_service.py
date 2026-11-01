"""PeopleService against real SQLite, with synthetic embeddings of known identities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings as hsettings
from hypothesis import strategies as st

from tests.conftest import FakeAnalyzer, count, make_photo, write_sql

DIM = 512


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Library:
    """Seeds faces directly into the database: `people` identities, each seen `per` times, plus strangers."""

    def __init__(self, app, people=4, per=6, strangers=3, noise=0.45, seed=0):
        self.app = app
        rng = np.random.default_rng(seed)
        self.centres = unit(rng.standard_normal((people, DIM)))
        self.truth: dict[int, int] = {}  # face id -> identity (-1 = stranger)
        vectors = [(i, self.centres[i] + noise * rng.standard_normal(DIM) / np.sqrt(DIM))
                   for i in range(people) for _ in range(per)]
        vectors += [(-1, rng.standard_normal(DIM)) for _ in range(strangers)]
        self.rng = rng

        def op(c):
            root = c.execute("INSERT INTO root(path) VALUES('lib')").lastrowid
            ids = []
            for n, (ident, v) in enumerate(vectors):
                m = c.execute("INSERT INTO media(root_id, rel_path, size, mtime, file_state) VALUES(?,?,1,1,'active')",
                              (root, f"{n}.jpg")).lastrowid
                item = c.execute("INSERT INTO item(media_id, width, height) VALUES(?, 100, 100)", (m,)).lastrowid
                f = c.execute(
                    "INSERT INTO face(item_id, x1, y1, x2, y2, landmarks, det_score, quality, detector_id) "
                    "VALUES(?, 0, 0, 50, 50, x'00', 0.9, 0.8, 'fake-detector')", (item,)).lastrowid
                c.execute("INSERT INTO face_embedding VALUES(?, ?, ?)",
                          (f, FakeAnalyzer.model_id, unit(v).astype(np.float32).tobytes()))
                ids.append((f, ident))
            return ids

        for f, ident in app._writer.submit(op):
            self.truth[f] = ident

    def faces_of(self, ident):
        return [f for f, i in self.truth.items() if i == ident]

    def state(self):
        """Everything a user decision can change, for exact before/after comparison."""
        from facegroup.adapters.storage.db import connect
        conn = connect(self.app.settings.db_path)
        try:
            faces = conn.execute("SELECT id, person_id, assigned_by FROM face ORDER BY id").fetchall()
            people = conn.execute("SELECT id, name, kind FROM person ORDER BY id").fetchall()
            rejects = conn.execute("SELECT face_id, person_id FROM face_reject ORDER BY 1, 2").fetchall()
            return [tuple(r) for r in faces], [tuple(r) for r in people], [tuple(r) for r in rejects]
        finally:
            conn.close()


@pytest.fixture
def lib(make_app):
    return Library(make_app())


def assigned(app, face_id):
    return count(app, "SELECT person_id FROM face WHERE id=?", face_id)


def by(app, face_id):
    from facegroup.adapters.storage.db import connect
    conn = connect(app.settings.db_path)
    try:
        return conn.execute("SELECT assigned_by FROM face WHERE id=?", (face_id,)).fetchone()[0]
    finally:
        conn.close()


def test_first_run_makes_pure_suggestions_and_assigns_nothing(lib):
    people = lib.app.people
    changed, groups = people.refresh()
    assert changed == 0  # nobody confirmed yet: nothing can be auto-assigned
    assert groups == 4
    for s in people.suggestions(sample=100):
        idents = {lib.truth[f] for f in people.suggestion_faces(s.id)}
        assert len(idents) == 1 and -1 not in idents  # each group is one real person; strangers stay unsorted


def test_accept_names_a_person_and_pulls_in_later_faces(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()[0]
    ident = lib.truth[people.suggestion_faces(s.id)[0]]
    pid = people.accept(s.id, "Mom")
    assert people.person(pid).name == "Mom"
    assert all(by(lib.app, f) == "user" for f in lib.faces_of(ident))
    assert people.suggestions() and all(
        lib.truth[f] != ident for g in people.suggestions() for f in people.suggestion_faces(g.id))
    # A new photo of the same person arrives: it is matched automatically.
    write_sql(lib.app, "UPDATE face SET person_id=NULL, assigned_by=NULL WHERE id=?", lib.faces_of(ident)[0])
    people.assign()
    assert assigned(lib.app, lib.faces_of(ident)[0]) == pid
    assert by(lib.app, lib.faces_of(ident)[0]) == "auto"


def test_exclude_faces_when_accepting(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()[0]
    faces = people.suggestion_faces(s.id)
    pid = people.accept(s.id, "Dad", exclude={faces[0]})
    assert by(lib.app, faces[0]) in (None, "auto")  # not confirmed by the user
    assert all(assigned(lib.app, f) == pid for f in faces[1:])


def test_reject_is_permanent_for_automatic_steps(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()[0]
    faces = people.suggestion_faces(s.id)
    pid = people.accept(s.id, "Ann", exclude={faces[0]})
    assert assigned(lib.app, faces[0]) == pid  # auto-matched (it really is Ann's face)
    people.reject([faces[0]], pid)
    for _ in range(2):
        people.refresh()
    assert assigned(lib.app, faces[0]) is None


def test_stale_automatic_vote_cannot_undo_a_rejection(lib):
    # The scheduler's assign() votes from a snapshot; the user rejects before its write lands.
    people = lib.app.people
    people.refresh()
    s = people.suggestions()[0]
    faces = people.suggestion_faces(s.id)
    pid = people.accept(s.id, "Ann", exclude={faces[0]})
    people.reject([faces[0]], pid)
    lib.app.people_store.write_auto_assignments([(faces[0], pid, 0.9)])  # the stale vote
    assert assigned(lib.app, faces[0]) is None


def test_merge_split_move_rename_and_undo_restore_exact_state(lib):
    people = lib.app.people
    people.refresh()
    sugg = people.suggestions()
    a = people.accept(sugg[0].id, "A")
    b = people.accept(sugg[1].id, "B")
    start = lib.state()

    people.rename(a, "Alice")
    assert people.person(a).name == "Alice"
    assert people.undo() == "rename"
    assert lib.state() == start

    people.merge(b, a)
    assert people.person(b) is None
    assert people.undo() == "merge"
    assert lib.state() == start

    faces_a = [f.id for f in people.person_faces(a)]
    new = people.split(faces_a[:2], "Alice (child)")
    assert {assigned(lib.app, f) for f in faces_a[:2]} == {new}
    assert people.undo() == "split"
    assert lib.state() == start and people.person(new) is None

    people.move(faces_a[:1], b)
    assert assigned(lib.app, faces_a[0]) == b
    assert people.undo() == "move"
    assert lib.state() == start

    people.reject(faces_a[:1], a)
    assert people.undo() == "reject"
    assert lib.state() == start


def test_merge_keeps_rejections(lib):
    people = lib.app.people
    people.refresh()
    sugg = people.suggestions()
    a = people.accept(sugg[0].id, "A")
    b = people.accept(sugg[1].id, "B")
    stranger = lib.faces_of(-1)[0]
    people.reject([stranger], b)  # "this face is not B"
    people.merge(b, a)  # B was a duplicate of A
    assert count(lib.app, "SELECT COUNT(*) FROM face_reject WHERE face_id=? AND person_id=?", stranger, a) == 1


def test_explicit_move_overrides_an_earlier_rejection_and_undo_restores_it(lib):
    people = lib.app.people
    people.refresh()
    a = people.accept(people.suggestions()[0].id, "A")
    face = [f.id for f in people.person_faces(a)][0]
    people.reject([face], a)
    before = lib.state()
    people.move([face], a)  # changed my mind: it is A after all
    assert assigned(lib.app, face) == a
    assert count(lib.app, "SELECT COUNT(*) FROM face_reject WHERE face_id=?", face) == 0
    people.undo()
    assert lib.state() == before


def test_merge_does_not_override_a_specific_rejection(lib):
    """Found by the property test: B contains a face the user said is not A; merging B into A must not
    put it in A. The face is left unassigned instead, and undo restores everything."""
    people = lib.app.people
    people.refresh()
    sugg = people.suggestions()
    a = people.accept(sugg[0].id, "A")
    b = people.accept(sugg[1].id, "B")
    face = [f.id for f in people.person_faces(b)][0]
    people.move([face], a)
    people.reject([face], a)  # "not A"
    people.move([face], b)  # it is B's
    assert count(lib.app, "SELECT COUNT(*) FROM face_reject WHERE face_id=? AND person_id=?", face, a) == 1
    before = lib.state()
    people.merge(b, a)
    assert assigned(lib.app, face) is None
    people.refresh()
    assert assigned(lib.app, face) != a
    people.undo()
    assert lib.state() == before


def test_combine_suggestions_makes_one_new_person(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()
    faces = people.suggestion_faces(s[0].id) + people.suggestion_faces(s[1].id)
    pid = people.combine([], [s[0].id, s[1].id], "")
    assert {assigned(lib.app, f) for f in faces} == {pid}
    assert all(by(lib.app, f) == "user" for f in faces)
    assert people.person(pid).name == ""  # unnamed: shown as "Face <id>"


def test_combine_people_and_suggestions_keeps_the_named_person_and_undoes_in_one_step(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()
    unnamed = people.accept(s[0].id, "")
    named = people.accept(s[1].id, "Mom")
    third = s[2].id
    before = lib.state()
    target = people.combine([unnamed, named], [third], "")
    assert target == named  # the named one survives
    assert people.person(unnamed) is None
    assert people.person(named).faces == sum(1 for f in lib.truth if assigned(lib.app, f) == named)
    assert people.undo() == "combine"  # a single undo reverses the whole combine
    assert lib.state() == before


def test_combine_can_rename(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()
    a, b = people.accept(s[0].id, ""), people.accept(s[1].id, "")
    target = people.combine([a, b], [], "Grandpa")
    assert people.person(target).name == "Grandpa"


def test_not_a_person_never_pulls_in_other_faces(lib):
    """User report: marking a few faces 'Not a person' seemed to mark the whole group. Marked faces must not vote."""
    people = lib.app.people
    people.refresh()
    faces = people.suggestion_faces(people.suggestions()[0].id)
    hid = people.not_a_person(face_ids=faces[:1])
    people.refresh()
    assert [f for f in lib.truth if assigned(lib.app, f) == hid] == faces[:1]


def test_not_a_person_goes_to_one_hidden_group(lib):
    people = lib.app.people
    people.refresh()
    sugg = people.suggestions()
    first = people.not_a_person(sugg[0].id)
    second = people.not_a_person(sugg[1].id)
    assert first == second
    assert all(p.id != first for p in people.people())  # hidden from the People list
    assert people.person(first).kind == "ignored"


def test_forget_deletes_faces_and_rescans_do_not_bring_them_back(make_app, library_dir):
    make_photo(library_dir / "a.jpg", color=(10, 20, 30))
    make_photo(library_dir / "b.jpg", color=(200, 20, 30))
    app = make_app()
    root = app.library.add_root(library_dir)
    app.library.scan(root)
    app.indexing.run_derive()
    app.indexing.run_faces()
    face = count(app, "SELECT MIN(id) FROM face")
    pid = app.people.split([face], "Someone")
    crop = app.cache.face_crop_path(face)
    assert crop.exists()
    assert app.people.forget(pid) == 1
    assert count(app, "SELECT COUNT(*) FROM face WHERE id=?", face) == 0
    assert not crop.exists()
    assert app.people.undo() is None  # forgetting cannot be undone
    # Re-run face analysis on everything: the forgotten face must not return; the other one does.
    write_sql(app, "UPDATE job SET state='pending', attempts=0 WHERE stage='faces'")
    app.indexing.run_faces()
    assert count(app, "SELECT COUNT(*) FROM face") == 1


def test_forget_keeps_automatic_matches_as_unsorted(lib):
    people = lib.app.people
    people.refresh()
    s = people.suggestions()[0]
    faces = people.suggestion_faces(s.id)
    pid = people.accept(s.id, "Ann", exclude={faces[0]})
    assert assigned(lib.app, faces[0]) == pid  # matched automatically, not confirmed
    assert people.forget(pid) == len(faces) - 1  # only the confirmed faces are deleted
    assert count(lib.app, "SELECT COUNT(*) FROM face WHERE id=?", faces[0]) == 1
    assert assigned(lib.app, faces[0]) is None
    assert count(lib.app, "SELECT COUNT(*) FROM suppressed_face") == len(faces) - 1


ACTIONS = st.lists(
    st.tuples(st.sampled_from(["accept", "reject", "move", "split", "merge", "combine", "undo", "refresh",
                               "not_person"]),
              st.integers(0, 1000), st.integers(0, 1000)),
    min_size=1, max_size=12,
)


@hsettings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(actions=ACTIONS)
def test_automatic_steps_never_change_a_user_decision(make_app, tmp_path, actions):
    """Property: after any sequence of user actions, assign() and suggest() leave every user-decided face as is."""
    app = make_app(data_dir=Path(tempfile.mkdtemp(dir=tmp_path)))  # fresh library per generated example
    lib = Library(app, people=3, per=4, strangers=2, seed=len(actions))
    people = app.people
    people.refresh()
    try:
        for kind, x, y in actions:
            sugg = people.suggestions()
            persons = [p.id for p in people.people()]
            faces = list(lib.truth)
            f = faces[x % len(faces)]
            if kind == "accept" and sugg:
                people.accept(sugg[x % len(sugg)].id, f"P{x}")
            elif kind == "not_person" and sugg:
                people.not_a_person(sugg[x % len(sugg)].id)
            elif kind == "reject" and persons:
                people.reject([f], persons[y % len(persons)])
            elif kind == "move" and persons:
                people.move([f], persons[y % len(persons)])
            elif kind == "split":
                people.split([f], "")
            elif kind == "merge" and len(persons) > 1:
                people.merge(persons[x % len(persons)], persons[y % len(persons)])
            elif kind == "combine":
                people.combine(persons[: (x % 3)], [g.id for g in sugg][: (y % 3)], "")
            elif kind == "undo":
                people.undo()
            elif kind == "refresh":
                people.refresh()

            before = [r for r in lib.state()[0] if r[2] == "user"]
            people.refresh()
            after = lib.state()[0]
            assert [r for r in after if r[2] == "user"] == before
            # Rejections always hold: no face sits in a person it was rejected from.
            rejected = people._store.rejections()
            for fid, pid, _ in after:
                assert pid not in rejected.get(fid, set())
    finally:
        app.close()
