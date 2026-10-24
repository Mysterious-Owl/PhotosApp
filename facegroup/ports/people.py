"""People ports: storage for people decisions, and the clustering algorithm used for suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class PeopleChange:
    """One user action, applied atomically and logged with enough state to undo it."""

    action: str  # accept, accept_into, ignore, rename, merge, move, split, reject, unassign
    face_ids: list[int] = field(default_factory=list)
    to_person: int | None = None  # existing target person
    new_person_name: str | None = None  # create a new target person with this name ('' = unnamed)
    new_person_kind: str = "person"  # person | ignored
    unassign: bool = False  # set person_id NULL instead of assigning
    keep_assigned_by: bool = False  # merge: faces keep auto/user status
    rejects: list[tuple[int, int]] = field(default_factory=list)  # (face_id, person_id) to add
    delete_persons: list[int] = field(default_factory=list)
    rename: tuple[int, str] | None = None
    clear_suggestion: int | None = None


@dataclass(frozen=True)
class PersonSummary:
    id: int
    name: str
    kind: str
    faces: int
    confirmed: int
    photos: int
    cover_face_id: int | None


@dataclass(frozen=True)
class SuggestionSummary:
    id: int
    size: int
    face_ids: list[int]  # best-quality first


@dataclass(frozen=True)
class FaceRow:
    id: int
    media_id: int
    person_id: int | None
    assigned_by: str | None
    quality: float


class Clusterer(Protocol):
    def cluster(self, vectors: np.ndarray) -> np.ndarray:
        """Labels per row; -1 = noise (stays unsorted)."""
        ...


class PeopleStore(Protocol):
    def confirmed_faces(self, model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(face_ids, person_ids, vectors) of faces the user confirmed."""
        ...

    def votable_faces(self, model_id: str) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
        """(face_ids, vectors, current person) of faces not decided by the user (unassigned or auto)."""
        ...

    def unassigned_faces(self, model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(face_ids, vectors, qualities) of faces with no person, excluding suppressed ones."""
        ...

    def rejections(self) -> dict[int, set[int]]: ...

    def write_auto_assignments(self, changes: list[tuple[int, int | None, float | None]]) -> int:
        """(face_id, person_id or None, score). Never touches faces the user decided. Returns rows changed."""
        ...

    def replace_suggestions(self, groups: list[list[int]]) -> None: ...

    def suggestions(self, sample: int = 8) -> list[SuggestionSummary]: ...

    def suggestion_faces(self, suggestion_id: int) -> list[int]: ...

    def people(self, include_hidden: bool = False) -> list[PersonSummary]: ...

    def person(self, person_id: int) -> PersonSummary | None: ...

    def person_faces(self, person_id: int, limit: int = 500, offset: int = 0) -> list[FaceRow]: ...

    def ignored_person_id(self) -> int | None: ...

    def apply_change(self, change: PeopleChange) -> int | None:
        """Apply atomically, log an undoable event, return the target (or new) person id."""
        ...

    def undo_last(self) -> str | None:
        """Undo the most recent not-yet-undone change; returns its action name, or None."""
        ...

    def forget_person(self, person_id: int, model_id: str) -> list[int]:
        """Delete the person and their faces; remember their embeddings so rescans skip them.
        Returns deleted face ids (for removing crops). Not undoable, by design."""
        ...

    def suppressed_vectors(self, model_id: str) -> np.ndarray: ...
