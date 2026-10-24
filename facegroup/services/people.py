"""PeopleService: who each face belongs to. Assignment, suggestions, and every user correction (with undo).

The user's decisions are constraints, never suggestions: automatic steps only touch faces nobody decided.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from facegroup.domain.people import AssignPolicy, knn_assign, merge_groups, prune_group, split_into_parts
from facegroup.ports.people import Clusterer, PeopleChange, PeopleStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuggestPolicy:
    min_quality: float = 0.15  # blurry/tiny faces make poor groups; they stay unsorted
    member_similarity: float = 0.40  # a member must average this similarity to the rest of its group
    merge_similarity: float = 0.55  # groups whose faces average this similarity are the same person
    min_group_size: int = 3


class PeopleService:
    def __init__(self, store: PeopleStore, clusterer: Clusterer, model_id: str,
                 assign_policy: AssignPolicy = AssignPolicy(), suggest_policy: SuggestPolicy = SuggestPolicy(),
                 on_forget: Callable[[list[int]], None] = lambda ids: None, part_similarity: float = 0.45) -> None:
        self._store = store
        self._clusterer = clusterer
        self._model_id = model_id
        self._assign = assign_policy
        self._suggest = suggest_policy
        self._on_forget = on_forget
        self._part_similarity = part_similarity

    # -- automatic steps ------------------------------------------------------------------------------

    def assign(self) -> int:
        """Re-vote every face nobody decided, against the faces the user confirmed. Returns changes."""
        ref_ids, ref_people, refs = self._store.confirmed_faces(self._model_id)
        face_ids, vectors, current = self._store.votable_faces(self._model_id)
        if len(face_ids) == 0:
            return 0
        votes = knn_assign(vectors, refs, ref_people, self._store.rejections(), self._assign,
                           query_ids=face_ids.tolist())
        changes = []
        for fid, now, vote in zip(face_ids.tolist(), current, votes):
            new = vote.person_id if vote else None
            if new != now:
                changes.append((fid, new, vote.score if vote else None))
        n = self._store.write_auto_assignments(changes)
        log.info("assign: %d of %d undecided faces changed", n, len(face_ids))
        return n

    def suggest(self) -> int:
        """Cluster the unassigned faces into suggested groups. Returns the number of groups."""
        face_ids, vectors, quality = self._store.unassigned_faces(self._model_id)
        keep = quality >= self._suggest.min_quality
        face_ids, vectors = face_ids[keep], vectors[keep]
        groups: list[list[int]] = []
        if len(face_ids) >= self._suggest.min_group_size:
            labels = self._clusterer.cluster(vectors)
            pure = []
            for label in sorted(set(labels.tolist()) - {-1}):
                idx = np.flatnonzero(labels == label)
                idx = idx[prune_group(vectors[idx], self._suggest.member_similarity)]
                if len(idx) >= 2:  # pairs may still merge into a bigger group below
                    pure.append(idx)
            for idx in merge_groups(pure, vectors, self._suggest.merge_similarity):
                if len(idx) >= self._suggest.min_group_size:
                    groups.append(face_ids[idx].tolist())
        groups.sort(key=len, reverse=True)
        self._store.replace_suggestions(groups)
        log.info("suggest: %d groups from %d unassigned faces", len(groups), len(face_ids))
        return len(groups)

    def refresh(self) -> tuple[int, int]:
        """After indexing or a correction: re-vote, then regroup what is still unassigned."""
        return self.assign(), self.suggest()

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

    def accept(self, suggestion_id: int, name: str = "", exclude: set[int] = frozenset()) -> int:
        faces = [f for f in self._store.suggestion_faces(suggestion_id) if f not in exclude]
        pid = self._store.apply_change(PeopleChange(
            "accept", face_ids=faces, new_person_name=name, clear_suggestions=[suggestion_id]))
        self.assign()
        return pid

    def accept_into(self, suggestion_id: int, person_id: int, exclude: set[int] = frozenset()) -> int:
        rejected = self._store.rejections()
        faces = [f for f in self._store.suggestion_faces(suggestion_id)
                 if f not in exclude and person_id not in rejected.get(f, ())]
        self._store.apply_change(PeopleChange(
            "accept_into", face_ids=faces, to_person=person_id, clear_suggestions=[suggestion_id]))
        self.assign()
        return person_id

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
        pid = self._store.apply_change(change)
        self.assign()
        return pid

    def rename(self, person_id: int, name: str) -> None:
        self._store.apply_change(PeopleChange("rename", to_person=person_id, rename=(person_id, name)))

    def merge(self, source_id: int, target_id: int) -> None:
        if source_id == target_id:
            return
        self._store.apply_change(PeopleChange("merge", to_person=target_id, delete_persons=[source_id]))
        self.assign()

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
        pid = self._store.apply_change(change)
        self.assign()
        return pid

    def confirm(self, face_ids: list[int], person_id: int) -> None:
        """'Yes, this is them': turns automatic matches into confirmed ones (which then vote)."""
        self._store.apply_change(PeopleChange("confirm", face_ids=face_ids, to_person=person_id))
        self.assign()

    def move(self, face_ids: list[int], person_id: int) -> None:
        self._store.apply_change(PeopleChange("move", face_ids=face_ids, to_person=person_id))
        self.assign()

    def split(self, face_ids: list[int], name: str = "") -> int:
        pid = self._store.apply_change(PeopleChange("split", face_ids=face_ids, new_person_name=name))
        self.assign()
        return pid

    def restore(self, face_ids: list[int]) -> None:
        """Out of 'Not a person' and back to unsorted; they will not be put back there automatically."""
        ignored = self._store.ignored_person_id()
        if ignored is not None:
            self.reject(face_ids, ignored)

    def reject(self, face_ids: list[int], person_id: int) -> None:
        """'Not this person': unassign and remember, so no automatic step puts them back."""
        self._store.apply_change(PeopleChange(
            "reject", face_ids=face_ids, unassign=True, rejects=[(f, person_id) for f in face_ids]))
        self.assign()

    def undo(self) -> str | None:
        action = self._store.undo_last()
        if action:
            self.assign()
        return action

    def forget(self, person_id: int) -> int:
        """Delete a person and their faces for good. Rescans will not bring them back. Not undoable."""
        ids = self._store.forget_person(person_id, self._model_id)
        self._on_forget(ids)
        return len(ids)
