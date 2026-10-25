import numpy as np

from facegroup.domain.people import AssignPolicy, knn_assign, merge_groups, prune_group, split_into_parts


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def identities(n_people=3, per=5, dim=64, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    centres = unit(rng.standard_normal((n_people, dim)))
    vecs, labels = [], []
    for p in range(n_people):
        for _ in range(per):
            vecs.append(centres[p] + noise * rng.standard_normal(dim) / np.sqrt(dim))
            labels.append(p + 100)
    return unit(np.array(vecs)), np.array(labels), centres


def test_assigns_to_the_person_most_neighbours_belong_to():
    refs, people, centres = identities()
    queries = unit(centres + 0.2 * np.random.default_rng(1).standard_normal(centres.shape) / 8)
    votes = knn_assign(queries, refs, people, {}, AssignPolicy(k=5, threshold=0.4, min_votes=2))
    assert [v.person_id for v in votes] == [100, 101, 102]
    assert all(0.4 <= v.score <= 1 for v in votes)


def test_stranger_is_not_assigned():
    refs, people, _ = identities()
    stranger = unit(np.random.default_rng(9).standard_normal((1, refs.shape[1])))
    assert knn_assign(stranger, refs, people, {}, AssignPolicy()) == [None]


def test_rejection_vetoes_the_winner():
    refs, people, centres = identities()
    q = centres[:1]
    assert knn_assign(q, refs, people, {7: {100}}, AssignPolicy(), query_ids=[7]) == [None]
    assert knn_assign(q, refs, people, {7: {101}}, AssignPolicy(), query_ids=[7])[0].person_id == 100


def test_min_votes_capped_by_how_many_confirmed_faces_exist():
    refs, people, centres = identities(n_people=1, per=1)
    vote = knn_assign(centres, refs, people, {}, AssignPolicy(min_votes=2))[0]
    assert vote is not None and vote.person_id == 100  # one confirmed face is enough when that is all there is


def test_no_strict_majority_means_no_assignment():
    # Two people equally close: a tie is not a decision.
    a, b = unit(np.eye(2)[0]), unit(np.eye(2)[1])
    refs = unit(np.array([a, a, b, b]))
    query = unit((a + b)[None])
    assert knn_assign(query, refs, np.array([1, 1, 2, 2]), {}, AssignPolicy(k=4, threshold=0.5)) == [None]


def test_empty_inputs():
    assert knn_assign(np.zeros((2, 8), np.float32), np.zeros((0, 8), np.float32), np.zeros(0, np.int64),
                      {}, AssignPolicy()) == [None, None]


def test_prune_group_drops_outliers():
    vecs, _, _ = identities(n_people=1, per=6)
    outlier = unit(np.random.default_rng(5).standard_normal((1, vecs.shape[1])))
    kept = prune_group(np.vstack([vecs, outlier]), 0.5)
    assert kept.tolist() == list(range(6))


def test_prune_catches_a_stranger_in_a_group_of_three():
    # Real-library failure: two faces of one person plus a stranger. The stranger is close to the group's
    # centre (which the pair dominates) but not to the other members.
    vecs, _, _ = identities(n_people=2, per=2, noise=0.2)
    group = np.vstack([vecs[0:2], vecs[2:3]])  # A, A, B
    assert prune_group(group, 0.4).tolist() == [0, 1]


def test_merge_groups_joins_split_person_but_not_strangers():
    vecs, labels, _ = identities(n_people=3, per=6)
    a1, a2 = np.flatnonzero(labels == 100)[:3], np.flatnonzero(labels == 100)[3:]
    b = np.flatnonzero(labels == 101)
    c = np.flatnonzero(labels == 102)
    merged = merge_groups([a1, b, a2, c], vecs, 0.55)
    assert sorted(sorted(set(labels[g].tolist())) for g in merged) == [[100], [101], [102]]
    assert sorted(len(g) for g in merged) == [6, 6, 6]


def test_split_into_parts_separates_a_mixed_group():
    vecs, labels, _ = identities(n_people=3, per=6)
    # A "mixed group": 6 faces of A, 2 of B, 1 of C, shuffled.
    pick = np.r_[np.flatnonzero(labels == 100), np.flatnonzero(labels == 101)[:2], np.flatnonzero(labels == 102)[:1]]
    rng = np.random.default_rng(3)
    pick = pick[rng.permutation(len(pick))]
    parts = split_into_parts(vecs[pick])
    got = [sorted(set(labels[pick][p].tolist())) for p in parts]
    assert got == [[100], [101], [102]]  # biggest part first, each part one person
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(pick)))  # every face exactly once


def test_split_into_parts_keeps_one_person_together():
    vecs, _, _ = identities(n_people=1, per=8)
    assert len(split_into_parts(vecs)) == 1
    assert split_into_parts(vecs[:0]) == []


# -- self-improvement rules ---------------------------------------------------------------------------------

from facegroup.domain.people import Vote, calibrate, dedupe_photos, one_per_photo  # noqa: E402


def test_one_per_photo_keeps_the_most_confident_vote():
    votes = [Vote(1, 0.6), Vote(1, 0.8), Vote(2, 0.5), None]
    items = [10, 10, 10, 11]
    assert one_per_photo(votes, items, set()) == [None, Vote(1, 0.8), Vote(2, 0.5), None]


def test_one_per_photo_respects_confirmed_faces():
    # Photo 10 already has a confirmed face of person 1: no automatic second "person 1" there.
    assert one_per_photo([Vote(1, 0.9)], [10], {(10, 1)}) == [None]


def test_dedupe_photos_keeps_one_face_per_photo():
    vecs, labels, _ = identities(n_people=2, per=4)
    idx = np.arange(5)  # 4 of A plus one B
    items = np.array([1, 2, 3, 3, 3])  # faces 2,3,4 share photo 3
    kept = dedupe_photos(idx, items, vecs)
    assert len(kept) == 3 and len(set(items[kept].tolist())) == 3
    assert 4 not in kept.tolist()  # the stranger (B) was the worst fit in the shared photo


def test_groups_sharing_a_photo_never_merge():
    vecs, labels, _ = identities(n_people=1, per=6)
    a, b = np.arange(3), np.arange(3, 6)
    items = np.array([1, 2, 3, 3, 4, 5])  # same photo appears in both groups
    assert len(merge_groups([a, b], vecs, 0.5, items=items)) == 2
    assert len(merge_groups([a, b], vecs, 0.5)) == 1  # without the photo rule they would merge


def test_leave_one_out_never_lets_a_face_vote_for_itself():
    vecs, labels, _ = identities(n_people=3, per=5)
    votes = knn_assign(vecs, vecs, labels, {}, AssignPolicy(k=5, threshold=0.4), leave_one_out=True)
    assert all(v and v.person_id == lab for v, lab in zip(votes, labels))
    lonely = knn_assign(vecs[:1], vecs[:1], labels[:1], {}, AssignPolicy(), leave_one_out=True)
    assert lonely == [None]  # its only reference is itself


def test_person_threshold_overrides_default():
    vecs, labels, centres = identities(n_people=1, per=5)
    q = centres
    strict = AssignPolicy(threshold=0.4, person_thresholds={100: 0.999})
    assert knn_assign(q, vecs, labels, {}, strict) == [None]
    loose = AssignPolicy(threshold=0.999, person_thresholds={100: 0.4})
    assert knn_assign(q, vecs, labels, {}, loose)[0].person_id == 100


def test_calibrate_picks_loosest_threshold_meeting_target():
    rng = np.random.default_rng(0)
    right = rng.uniform(0.45, 0.9, 300)  # correct matches score high
    wrong = rng.uniform(0.30, 0.50, 60)  # mistakes cluster low
    scores = np.r_[right, wrong]
    correct = np.r_[np.ones(300), np.zeros(60)]
    c = calibrate(scores, correct, target=0.98, min_samples=30)
    assert c.threshold is not None and 0.42 <= c.threshold <= 0.52
    assert c.precision >= 0.98
    assert calibrate(scores[:10], correct[:10], 0.98, min_samples=30).threshold is None  # not enough evidence
