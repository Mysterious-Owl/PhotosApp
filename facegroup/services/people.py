"""PeopleService: who each face belongs to. Assignment, suggestions, and every user correction (with undo).

The user's decisions are constraints, never suggestions: automatic steps only touch faces nobody decided.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np

from facegroup.domain.people import (AssignPolicy, calibrate, dedupe_photos, knn_assign, merge_groups,
                                     one_per_photo, prune_group, split_into_parts)
from facegroup.ports.people import Clusterer, PeopleChange, PeopleStore

log = logging.getLogger(__name__)


def _serialized(method):
    """One people operation at a time. The scheduler's refresh reads a snapshot and writes it back later; without
    this, a correction made in between (such as "Not this person") could be overwritten by that stale snapshot."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


@dataclass(frozen=True)
class SuggestPolicy:
    min_quality: float = 0.15  # blurry/tiny faces make poor groups; they stay unsorted
    member_similarity: float = 0.40  # a member must average this similarity to the rest of its group
    merge_similarity: float = 0.55  # groups whose faces average this similarity are the same person
    min_group_size: int = 3


@dataclass(frozen=True)
class LearnPolicy:
    target_precision: float = 0.98  # automatic matches must be right this often (measured on your own labels)
    min_samples: int = 30  # evidence needed before strictness is learned (enough to make it STRICTER)
    min_samples_to_loosen: int = 200  # ...but going LOOSER than the default needs much more evidence
    min_person_samples: int = 10  # ... and before a person gets their own
    floor: float = 0.35  # never looser than this, whatever the data says
    ceiling: float = 0.75  # never stricter than this
    person_below: float = 0.08  # a person's strictness stays within [global - below, global + above]
    person_above: float = 0.15
    question_margin: float = 0.06  # automatic matches this close to the line are worth asking about
    self_test_size: int = 2500  # confirmed faces replayed per self-test (spread across people); keeps it < 1 s
    retest_change: float = 0.02  # re-run the self-test when confirmed faces change by 2% ...
    retest_feedback: int = 10  # ... or this many more automatic matches were judged
    same_person_min: float = 0.40  # people/groups this alike are worth a "same person?" question


@dataclass
class Question:
    kind: str  # is_person | same_person | group_is
    score: float  # how close / how alike: used for ordering
    face_id: int | None = None
    person_id: int | None = None
    other_id: int | None = None
    suggestion_id: int | None = None
    face_ids: list[int] = field(default_factory=list)


class PeopleService:
    def __init__(self, store: PeopleStore, clusterer: Clusterer, model_id: str,
                 assign_policy: AssignPolicy = AssignPolicy(), suggest_policy: SuggestPolicy = SuggestPolicy(),
                 on_forget: Callable[[list[int]], None] = lambda ids: None, part_similarity: float = 0.45,
                 learn_policy: LearnPolicy = LearnPolicy()) -> None:
        self._store = store
        self._clusterer = clusterer
        self._model_id = model_id
        self._assign = assign_policy
        self._suggest = suggest_policy
        self._on_forget = on_forget
        self._part_similarity = part_similarity
        self._learn = learn_policy
        self._policy_cache: tuple[tuple, AssignPolicy] | None = None
        self._lock = threading.RLock()  # re-entrant: user actions call assign()

    # -- automatic steps ------------------------------------------------------------------------------

    @_serialized
    def assign(self) -> int:
        """Re-vote every face nobody decided, against the faces the user confirmed. Returns changes.

        Uses the strictness learned from the user's own labels (see `policy`) and the rule that two faces in one
        photo are two different people."""
        ref_ids, ref_people, refs, ref_items = self._store.confirmed_faces(self._model_id)
        face_ids, vectors, current, items = self._store.votable_faces(self._model_id)
        policy = self.policy(ref_people, refs)
        if len(face_ids) == 0:
            return 0
        votes = knn_assign(vectors, refs, ref_people, self._store.rejections(), policy,
                           query_ids=face_ids.tolist())
        votes = one_per_photo(votes, items, set(zip(ref_items.tolist(), ref_people.tolist())))
        changes = []
        for fid, now, vote in zip(face_ids.tolist(), current, votes):
            new = vote.person_id if vote else None
            if new != now:
                changes.append((fid, new, vote.score if vote else None))
        n = self._store.write_auto_assignments(changes)
        log.info("assign: %d of %d undecided faces changed", n, len(face_ids))
        return n

    @_serialized
    def suggest(self) -> int:
        """Cluster the unassigned faces into suggested groups. Returns the number of groups."""
        face_ids, vectors, quality, items = self._store.unassigned_faces(self._model_id)
        keep = quality >= self._suggest.min_quality
        face_ids, vectors, items = face_ids[keep], vectors[keep], items[keep]
        groups: list[list[int]] = []
        if len(face_ids) >= self._suggest.min_group_size:
            labels = self._clusterer.cluster(vectors)
            pure = []
            for label in sorted(set(labels.tolist()) - {-1}):
                idx = np.flatnonzero(labels == label)
                idx = dedupe_photos(idx, items, vectors)  # one face per photo: two in one photo are two people
                idx = idx[prune_group(vectors[idx], self._suggest.member_similarity)]
                if len(idx) >= 2:  # pairs may still merge into a bigger group below
                    pure.append(idx)
            for idx in merge_groups(pure, vectors, self._suggest.merge_similarity, items=items):
                if len(idx) >= self._suggest.min_group_size:
                    groups.append(face_ids[idx].tolist())
        groups.sort(key=len, reverse=True)
        self._store.replace_suggestions(groups)
        log.info("suggest: %d groups from %d unassigned faces", len(groups), len(face_ids))
        return len(groups)

    @_serialized
    def refresh(self) -> tuple[int, int]:
        """After indexing or a correction: re-vote, then regroup what is still unassigned."""
        return self.assign(), self.suggest()

    # -- self-improvement: learned strictness ----------------------------------------------------------------

    @_serialized
    def policy(self, ref_people: np.ndarray | None = None, refs: np.ndarray | None = None) -> AssignPolicy:
        """The matching strictness, learned from the user's labels when there is enough evidence.

        Self-test: each confirmed face is hidden in turn and matched against the others ("who would I have said
        this is?"), and every judged automatic match is added. The loosest strictness that keeps measured
        precision at the target is chosen, globally and per person. Cached until the labels change."""
        if refs is None:
            _, ref_people, refs, _ = self._store.confirmed_faces(self._model_id)
        feedback = self._store.feedback()
        lp, default = self._learn, self._assign
        if self._policy_cache:
            (n_refs, n_fb), cached = self._policy_cache
            small_change = abs(len(refs) - n_refs) <= max(2, lp.retest_change * n_refs)
            if small_change and len(feedback) - n_fb < lp.retest_feedback and (n_refs > 0) == (len(refs) > 0):
                return cached
        probe = replace(default, threshold=lp.floor - 0.05, person_thresholds=None)
        sample = _spread_sample(ref_people, lp.self_test_size)
        loo = (knn_assign(refs[sample], refs, ref_people, None, probe, exclude=sample)
               if len(refs) > 1 else [])
        scores, correct, pred = [], [], []
        for vote, truth in zip(loo, ref_people[sample].tolist()):
            if vote:
                scores.append(vote.score)
                correct.append(vote.person_id == truth)
                pred.append(vote.person_id)
        for score, ok, person in feedback:
            scores.append(score)
            correct.append(bool(ok))
            pred.append(person)
        scores_a, correct_a, pred_a = np.array(scores), np.array(correct, bool), np.array(pred)
        overall = calibrate(scores_a, correct_a, lp.target_precision, lp.min_samples)
        if overall.threshold is not None:
            g, source = float(np.clip(overall.threshold, lp.floor, lp.ceiling)), "learned"
            if g < default.threshold and len(scores) < lp.min_samples_to_loosen:
                g = default.threshold  # careful by default: "100% right" on a few dozen examples proves little
        elif len(scores) >= lp.min_samples:
            g, source = lp.ceiling, "strict"  # plenty of evidence, target never met: be as careful as allowed
        else:
            g, source = default.threshold, "default"
        per: dict[int, float] = {}
        if source != "default":
            lo, hi = max(lp.floor, g - lp.person_below), min(lp.ceiling, g + lp.person_above)
            if len(scores) < lp.min_samples_to_loosen:
                lo = max(lo, default.threshold)  # same rule per person
            for person in set(pred):
                sel = pred_a == person
                if sel.sum() < lp.min_person_samples:
                    continue
                c = calibrate(scores_a[sel], correct_a[sel], lp.target_precision, lp.min_person_samples)
                per[int(person)] = float(np.clip(c.threshold, lo, hi)) if c.threshold is not None else hi
        policy = replace(default, threshold=round(g, 3), person_thresholds=per or None)
        self._policy_cache = ((len(refs), len(feedback)), policy)
        self._store.set_setting("calibration", {
            "threshold": round(g, 3), "source": source, "precision": overall.precision,
            "samples": len(scores), "self_test": len([v for v in loo if v]), "judged": len(feedback),
            "per_person": {str(k): round(v, 3) for k, v in per.items()}, "target": lp.target_precision,
            "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        log.info("policy: threshold %.3f (%s) from %d samples, %d per-person", g, source, len(scores), len(per))
        return policy

    @_serialized
    def learning_stats(self) -> dict:
        """What FaceGroup has learned, for the Check page."""
        self.policy()
        cal = self._store.get_setting("calibration") or {}
        fb = self._store.feedback()
        recent = fb[-200:]
        cal["judged_recent"] = len(recent)
        cal["accuracy_recent"] = round(sum(ok for _, ok, _ in recent) / len(recent), 4) if recent else None
        cal["auto_matches"] = len(self._store.auto_assignments())
        return cal

    # -- self-improvement: the most useful questions ---------------------------------------------------------

    def questions(self, per_kind: int = 6) -> list[Question]:
        """Questions whose answers teach the most: groups that look like a known person, people that look like
        each other, and automatic matches FaceGroup is least sure about."""
        _, ref_people, refs, ref_items = self._store.confirmed_faces(self._model_id)
        policy = self.policy(ref_people, refs)
        lp = self._learn
        out: list[Question] = []
        people = sorted(set(ref_people.tolist()))
        sums = {p: refs[ref_people == p].sum(axis=0) for p in people}
        counts = {p: int((ref_people == p).sum()) for p in people}
        rejected = self._store.rejections()

        group_qs = []
        for s in self._store.suggestions(sample=8):
            faces = self._store.suggestion_faces(s.id)
            ids, vecs = self._store.face_vectors(faces, self._model_id)
            if not len(ids) or not people:
                continue
            gsum = vecs.sum(axis=0)
            best = max(people, key=lambda p: float(sums[p] @ gsum) / (counts[p] * len(ids)))
            sim = float(sums[best] @ gsum) / (counts[best] * len(ids))
            if all(best in rejected.get(f, ()) for f in ids):
                continue  # already answered "no"
            if sim >= max(lp.same_person_min, policy.for_person(best) - lp.question_margin):
                group_qs.append(Question("group_is", sim, person_id=best, suggestion_id=s.id, face_ids=s.face_ids))
        out += sorted(group_qs, key=lambda q: -q.score)[:per_kind]

        distinct = self._store.distinct_pairs()
        photos = {p: set(ref_items[ref_people == p].tolist()) for p in people}
        pair_qs = []
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                if (a, b) in distinct or photos[a] & photos[b]:
                    continue  # answered already, or seen together in a photo: certainly two people
                sim = float(sums[a] @ sums[b]) / (counts[a] * counts[b])
                if sim >= lp.same_person_min:
                    pair_qs.append(Question("same_person", sim, person_id=a, other_id=b))
        out += sorted(pair_qs, key=lambda q: -q.score)[:per_kind]

        face_qs = []
        for face_id, person, score, _ in self._store.auto_assignments():
            margin = score - policy.for_person(person)
            if margin < lp.question_margin:
                face_qs.append(Question("is_person", margin, face_id=face_id, person_id=person))
        out += sorted(face_qs, key=lambda q: q.score)[:per_kind]
        return out

    @_serialized
    def answer_is_person(self, face_id: int, person_id: int, yes: bool) -> None:
        if yes:
            self.confirm([face_id], person_id)
        else:
            self.reject([face_id], person_id)

    @_serialized
    def answer_same_person(self, a: int, b: int, yes: bool) -> int | None:
        if yes:
            return self.combine([a, b], [])
        self._store.mark_distinct(a, b)
        return None

    @_serialized
    def answer_group_is(self, suggestion_id: int, person_id: int, yes: bool) -> None:
        if yes:
            self.accept_into(suggestion_id, person_id)
            return
        faces = self._store.suggestion_faces(suggestion_id)
        self._store.apply_change(PeopleChange("not_group", rejects=[(f, person_id) for f in faces]))

    # -- reading ----------------------------------------------------------------------------------------

    def people(self):
        return self._store.people()

    def person(self, person_id: int):
        return self._store.person(person_id)

    def person_faces(self, person_id: int, limit: int = 500, offset: int = 0):
        return self._store.person_faces(person_id, limit, offset)

    def suggestions(self, sample: int = 8):
        return self._store.suggestions(sample)

    def suggestion_faces(self, suggestion_id: int) -> list[int]:
        return self._store.suggestion_faces(suggestion_id)

    def parts(self, face_ids: list[int]) -> list[list[int]]:
        """The faces split into parts that each look like one person, biggest first (for breaking up a mixed
        group). Faces without an embedding come last, as their own part."""
        ids, vectors = self._store.face_vectors(face_ids, self._model_id)
        parts = [[ids[i] for i in part] for part in split_into_parts(vectors, self._part_similarity)]
        missing = [f for f in face_ids if f not in set(ids)]
        return parts + ([missing] if missing else [])

    def hidden(self):
        """The 'Not a person' group, if it exists."""
        pid = self._store.ignored_person_id()
        return self._store.person(pid) if pid else None

    # -- user decisions (each one atomic, logged, undoable) -----------------------------------------------

    @_serialized
    def accept(self, suggestion_id: int, name: str = "", exclude: set[int] = frozenset()) -> int:
        faces = [f for f in self._store.suggestion_faces(suggestion_id) if f not in exclude]
        return self._apply(PeopleChange(
            "accept", face_ids=faces, new_person_name=name, clear_suggestions=[suggestion_id]))

    @_serialized
    def accept_into(self, suggestion_id: int, person_id: int, exclude: set[int] = frozenset()) -> int:
        rejected = self._store.rejections()
        faces = [f for f in self._store.suggestion_faces(suggestion_id)
                 if f not in exclude and person_id not in rejected.get(f, ())]
        self._apply(PeopleChange("accept_into", face_ids=faces, to_person=person_id, clear_suggestions=[suggestion_id]))
        return person_id

    @_serialized
    def not_a_person(self, suggestion_id: int | None = None, face_ids: list[int] | None = None) -> int:
        """Cartoons, statues, animals. Only these faces go to the hidden 'Not a person' group: nothing is added to it
        automatically, because a wrong mark would otherwise drag a real person's look-alikes along."""
        faces = list(face_ids or []) + (self._store.suggestion_faces(suggestion_id) if suggestion_id else [])
        ignored = self._store.ignored_person_id()
        change = PeopleChange("ignore", face_ids=faces, clear_suggestions=[suggestion_id] if suggestion_id else [])
        if ignored is None:
            change.new_person_name, change.new_person_kind = "Not a person", "ignored"
        else:
            change.to_person = ignored
        return self._apply(change)

    @_serialized
    def rename(self, person_id: int, name: str) -> None:
        self._store.apply_change(PeopleChange("rename", to_person=person_id, rename=(person_id, name)))

    @_serialized
    def merge(self, source_id: int, target_id: int) -> None:
        if source_id == target_id:
            return
        # reset_policy: per-person strictness is keyed by person id, and one id just went away
        self._apply(PeopleChange("merge", to_person=target_id, delete_persons=[source_id]), reset_policy=True)

    @_serialized
    def combine(self, person_ids: list[int], suggestion_ids: list[int], name: str = "") -> int | None:
        """One person from several people and/or suggested groups, in one undoable step.

        The target is the existing person with a name (or, failing that, the most faces); the others merge into
        it. Faces from suggestions become confirmed. With only suggestions, a new person is created.
        """
        people = [p for p in (self._store.person(i) for i in dict.fromkeys(person_ids)) if p and p.kind == "person"]
        rejected = self._store.rejections()
        faces: list[int] = []
        for sid in dict.fromkeys(suggestion_ids):
            faces += self._store.suggestion_faces(sid)
        if not people and not faces:
            return None
        change = PeopleChange("combine", clear_suggestions=list(suggestion_ids))
        if people:
            target = max(people, key=lambda p: (bool(p.name), p.faces, -p.id))
            change.to_person = target.id
            change.delete_persons = [p.id for p in people if p.id != target.id]
            change.face_ids = [f for f in faces if target.id not in rejected.get(f, ())]
            if name.strip():
                change.rename = (target.id, name)
        else:
            change.face_ids = faces
            change.new_person_name = name
        return self._apply(change, reset_policy=True)  # as in merge

    @_serialized
    def confirm(self, face_ids: list[int], person_id: int) -> None:
        """'Yes, this is them': turns automatic matches into confirmed ones (which then vote)."""
        self._apply(PeopleChange("confirm", face_ids=face_ids, to_person=person_id))

    @_serialized
    def move(self, face_ids: list[int], person_id: int) -> None:
        self._apply(PeopleChange("move", face_ids=face_ids, to_person=person_id))

    @_serialized
    def split(self, face_ids: list[int], name: str = "") -> int:
        return self._apply(PeopleChange("split", face_ids=face_ids, new_person_name=name))

    @_serialized
    def restore(self, face_ids: list[int]) -> None:
        """Out of 'Not a person' and back to unsorted; they will not be put back there automatically."""
        ignored = self._store.ignored_person_id()
        if ignored is not None:
            self.reject(face_ids, ignored)

    @_serialized
    def reject(self, face_ids: list[int], person_id: int) -> None:
        """'Not this person': unassign and remember, so no automatic step puts them back."""
        self._apply(PeopleChange("reject", face_ids=face_ids, unassign=True, rejects=[(f, person_id) for f in face_ids]))

    @_serialized
    def undo(self) -> str | None:
        action = self._store.undo_last()
        if action:
            self.assign()
        return action

    @_serialized
    def forget(self, person_id: int) -> int:
        """Delete a person and the faces the user confirmed as them, for good. Rescans will not bring those back.
        Automatic matches are only unassigned (they may be someone else) and get re-voted. Not undoable."""
        ids = self._store.forget_person(person_id, self._model_id)
        self._on_forget(ids)
        self._policy_cache = None  # as in merge
        self.assign()
        return len(ids)

    def _apply(self, change: PeopleChange, reset_policy: bool = False):
        """Apply one user decision, then re-vote the undecided faces. Callers hold the lock (via _serialized).
        reset_policy: the change removed person ids, so the cached per-person strictness is stale."""
        result = self._store.apply_change(change)
        if reset_policy:
            self._policy_cache = None
        self.assign()
        return result


def _spread_sample(people: np.ndarray, size: int) -> np.ndarray:
    """Indices of up to `size` faces, spread evenly over people (so one big family album cannot dominate),
    deterministic so repeated self-tests agree."""
    if len(people) <= size:
        return np.arange(len(people))
    rng = np.random.default_rng(0)
    groups = {p: np.flatnonzero(people == p) for p in np.unique(people)}
    per = max(1, size // len(groups))
    picked = np.concatenate([rng.permutation(idx)[:per] for idx in groups.values()])
    if len(picked) > size:  # more people than the sample holds: a random subset, not the lowest ids
        picked = rng.choice(picked, size, replace=False)
    return np.sort(picked)
