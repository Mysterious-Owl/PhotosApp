"""Search ports: the image-text embedder, and the stores search reads from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Maps images and text queries into one space; vectors are L2-normalised, so dot product = cosine."""

    model_id: str
    dim: int

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SearchFilters:
    person_ids: tuple[int, ...] = ()  # all of these people must be in the photo
    date_from: str | None = None  # ISO date, inclusive
    date_to: str | None = None
    root_id: int | None = None
    folder: str | None = None  # path prefix inside the root
    kind: str | None = None  # "photos" = hide clutter; or one category (screenshot, document, receipt, meme, ...)
    favorites: bool = False
    min_rating: int | None = None
    show_hidden: bool = False


@dataclass(frozen=True)
class SearchHit:
    media_id: int
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    stack_size: int = 1  # > 1: similar shots folded behind this one
    text_rank: int | None = None  # rank among photos whose text (OCR) matched


@dataclass
class SearchResult:
    hits: list[SearchHit] = field(default_factory=list)
    candidates: int = 0  # photos left after filters
    used_vectors: bool = False
    missing_embeddings: int = 0  # candidates not yet embedded (indexing still running)
    people: list[int] = field(default_factory=list)  # people recognised in the query text


class SearchStore(Protocol):
    def candidate_items(self, f: SearchFilters) -> tuple[np.ndarray, np.ndarray]:
        """(item_ids, media_ids) of active photos matching the filters."""
        ...

    def item_vectors(self, model_id: str, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(item_ids that have an embedding, vectors) for the given items."""
        ...

    def keyword_media(self, query: str, media_ids: np.ndarray | None, limit: int) -> list[int]:
        """Media ids whose indexed text matches, best first."""
        ...

    def media_vector(self, media_id: int, model_id: str) -> np.ndarray | None: ...

    def text_media(self, query: str, media_ids: np.ndarray | None, limit: int) -> list[int]:
        """Photos whose text found in the image (OCR) matches, best first."""
        ...

    def newest_first(self, media_ids: np.ndarray) -> list[int]: ...

    def stack_map(self, media_ids: list[int]) -> dict[int, tuple[int, bool, int]]: ...
