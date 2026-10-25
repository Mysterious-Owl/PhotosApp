"""Self-improvement: same-photo rule, scorecard, learned strictness, and the Quick check questions."""

import numpy as np
import pytest

from tests.conftest import count, write_sql
from tests.unit.test_people_service import Library, assigned, by


@pytest.fixture
def lib(make_app):
    return Library(make_app(), people=4, per=8, strangers=3, seed=1)


def put_in_same_photo(lib, a, b):
    item = count(lib.app, "SELECT item_id FROM face WHERE id=?", a)
    write_sql(lib.app, "UPDATE face SET item_id=? WHERE id=?", item, b)


# -- 1. two faces in one photo are two people -----------------------------------------------------------------

def test_two_faces_in_one_photo_are_never_both_matched_to_one_person(lib):
    people = lib.app.people
    faces = lib.faces_of(0)
    pid = people.split(faces[:4], "A")
    put_in_same_photo(lib, faces[4], faces[5])  # two A-looking faces in one photo (e.g. a lookalike sibling)
    people.assign()
    owners = [assigned(lib.app, f) for f in faces[4:6]]
    assert owners.count(pid) == 1


def test_a_confirmed_face_blocks_a_second_match_in_its_photo(lib):
    people = lib.app.people
    faces = lib.faces_of(0)
    pid = people.split(faces[:4], "A")
    put_in_same_photo(lib, faces[0], faces[6])  # photo already has a confirmed A
    people.assign()
    assert assigned(lib.app, faces[6]) != pid


def test_suggested_groups_never_hold_two_faces_from_one_photo(lib):
    faces = lib.faces_of(1)
    put_in_same_photo(lib, faces[0], faces[1])
    put_in_same_photo(lib, faces[0], faces[2])
    lib.app.people.suggest()
    for s in lib.app.people.suggestions(sample=100):
        items = [count(lib.app, "SELECT item_id FROM face WHERE id=?", f) for f in lib.app.people.suggestion_faces(s.id)]
        assert len(items) == len(set(items))


# -- 3. scorecard and learned strictness ------------------------------------------------------------------------

def test_scorecard_records_judged_automatic_matches_and_undo_removes_them(lib):
    people = lib.app.people
    faces = lib.faces_of(0)
    pid = people.split(faces[:4], "A")
    auto = [f for f in faces[4:] if by(lib.app, f) == "auto"]
    assert len(auto) >= 2
    people.confirm([auto[0]], pid)
    people.reject([auto[1]], pid)
    rows = count(lib.app, "SELECT COUNT(*) FROM assign_feedback")
    assert rows == 2
    assert count(lib.app, "SELECT correct FROM assign_feedback WHERE face_id=?", auto[0]) == 1
    assert count(lib.app, "SELECT correct FROM assign_feedback WHERE face_id=?", auto[1]) == 0
    people.undo()
    assert count(lib.app, "SELECT COUNT(*) FROM assign_feedback") == 1


def test_strictness_stays_default_without_enough_labels(lib):
    people = lib.app.people
    people.split(lib.faces_of(0)[:3], "A")
    stats = people.learning_stats()
    assert stats["source"] == "default" and stats["threshold"] == pytest.approx(0.42)


def test_strictness_is_learned_once_there_are_enough_labels(make_app):
    lib = Library(make_app(), people=6, per=10, strangers=0, noise=0.45, seed=2)
    people = lib.app.people
    for i in range(6):
        people.split(lib.faces_of(i), f"P{i}")
    stats = people.learning_stats()
    assert stats["source"] == "learned"
    assert 0.35 <= stats["threshold"] <= 0.75
    assert stats["self_test"] >= 30


def test_mistakes_make_it_stricter(make_app):
    lib = Library(make_app(), people=6, per=10, strangers=0, seed=3)
    people = lib.app.people
    for i in range(6):
        people.split(lib.faces_of(i), f"P{i}")
    before = people.policy().threshold
    # The user rejected many confident automatic matches: precision at those scores is clearly too low.
    event = lib.app._writer.submit(lambda c: c.execute(
        "INSERT INTO people_event(action, payload_json, at) VALUES('test', '{}', 'x')").lastrowid)
    rng = np.random.default_rng(0)
    for s in rng.uniform(before, before + 0.12, 80):
        write_sql(lib.app, "INSERT INTO assign_feedback(event_id, face_id, person_id, score, correct, at) "
                           "VALUES(?, 0, ?, ?, 0, 'x')", event, int(count(lib.app, "SELECT MIN(id) FROM person")), float(s))
    assert people.policy().threshold > before


# -- 2. questions ---------------------------------------------------------------------------------------------------

def test_question_a_group_that_looks_like_a_known_person(lib):
    people = lib.app.people
    faces = lib.faces_of(0)
    pid = people.split(faces[:3], "A")
    # Leave the rest of A unassigned in a suggestion (as if FaceGroup were unsure).
    write_sql(lib.app, "UPDATE face SET person_id=NULL, assigned_by=NULL WHERE assigned_by='auto'")
    lib.app.people_store.replace_suggestions([faces[3:]])
    qs = [q for q in people.questions() if q.kind == "group_is"]
    assert qs and qs[0].person_id == pid
    people.answer_group_is(qs[0].suggestion_id, pid, yes=False)
    assert not [q for q in people.questions() if q.kind == "group_is" and q.person_id == pid]
    assert count(lib.app, "SELECT COUNT(*) FROM face_reject WHERE person_id=?", pid) == len(faces[3:])


def test_question_two_people_that_look_alike_then_combine_or_keep_apart(make_app):
    lib = Library(make_app(), people=3, per=8, strangers=0, seed=4)
    people = lib.app.people
    faces = lib.faces_of(0)
    a, b = people.split(faces[:4], "A"), people.split(faces[4:], "")
    qs = [q for q in people.questions() if q.kind == "same_person"]
    assert qs and {qs[0].person_id, qs[0].other_id} == {a, b}
    people.answer_same_person(a, b, yes=False)
    assert not [q for q in people.questions() if q.kind == "same_person"]  # never asked again
    c = people.split(lib.faces_of(1)[:4], "")
    d = people.split(lib.faces_of(1)[4:], "")
    q = [q for q in people.questions() if q.kind == "same_person"][0]
    survivor = people.answer_same_person(q.person_id, q.other_id, yes=True)
    assert survivor in (c, d) and people.person(c if survivor == d else d) is None


def test_question_borderline_automatic_match(lib):
    people = lib.app.people
    faces = lib.faces_of(0)
    pid = people.split(faces[:4], "A")
    write_sql(lib.app, "UPDATE face SET assign_score=? WHERE assigned_by='auto' AND person_id=?",
              people.policy().for_person(pid) + 0.01, pid)
    qs = [q for q in people.questions() if q.kind == "is_person"]
    assert qs and all(q.person_id == pid for q in qs)
    people.answer_is_person(qs[0].face_id, pid, yes=True)
    assert by(lib.app, qs[0].face_id) == "user"


def test_loosening_needs_much_more_evidence_than_tightening(make_app):
    """A few dozen perfect examples must not make FaceGroup looser than its default."""
    lib = Library(make_app(), people=4, per=10, strangers=0, noise=0.2, seed=5)  # easy: very distinct people
    people = lib.app.people
    for i in range(4):
        people.split(lib.faces_of(i), f"P{i}")
    stats = people.learning_stats()
    assert stats["source"] == "learned" and stats["samples"] < 200
    assert stats["threshold"] >= 0.42  # not loosened on 40 examples
    assert all(t >= 0.42 for t in stats["per_person"].values())
