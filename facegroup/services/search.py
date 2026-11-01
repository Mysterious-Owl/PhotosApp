"""SearchService: filter first, then rank by meaning (embeddings) and by words (file and folder names, text read
from images); people's names in the query become a person filter.

Never writes. The query encoder (text side of the embedder) is loaded on first use and kept.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from dataclasses import replace

from facegroup.domain.search import parse_people, reciprocal_rank_fusion, top_media
from facegroup.ports.search import Embedder, SearchFilters, SearchHit, SearchResult, SearchStore

log = logging.getLogger(__name__)


class SearchService:
    def __init__(self, store: SearchStore, embedder_factory: Callable[[], Embedder], model_id: str,
                 limit: int = 200, people_names: Callable[[], list[tuple[int, str]]] = lambda: []) -> None:
        self._store = store
        self._make_embedder = embedder_factory
        self._model_id = model_id
        self._limit = limit
        self._people_names = people_names
        self._embedder: Embedder | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    def _query_encoder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = self._make_embedder()
            return self._embedder

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Text side of the search model, shared with OrganizeService (category descriptions)."""
        return self._query_encoder().embed_queries(texts)

    def warm_up(self) -> None:
        """Load the query encoder ahead of the first search (it takes a few seconds)."""
        if self._model_id:
            self._query_encoder().embed_queries(["warm up"])

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.close()
            self._embedder = None

    def search(self, text: str, filters: SearchFilters = SearchFilters(), parse_names: bool = True) -> SearchResult:
        # "Mom and Dad beach" -> people Mom + Dad, text "beach". parse_names=False searches the words literally
        # (for names that are also ordinary words, e.g. "Rose").
        named, text = parse_people(text, self._people_names()) if parse_names else ([], text)
        if named:
            filters = replace(filters, person_ids=tuple(dict.fromkeys((*filters.person_ids, *named))))
        items, item_media = self._store.candidate_items(filters)
        result = SearchResult(candidates=len(set(item_media.tolist())), people=named)
        text = text.strip()
        if not text:  # filters only: newest first
            media = self._store.newest_first(np.unique(item_media))
            result.hits = self._fold_stacks([SearchHit(m, 0.0) for m in media])[: self._limit]
            return result

        rankings: list[list[int]] = []
        vector_rank: dict[int, int] = {}
        ids, vectors = self._store.item_vectors(self._model_id, items)
        result.missing_embeddings = len(items) - len(ids)
        if len(ids):
            query = self._query_encoder().embed_queries([text])[0]
            media_of = dict(zip(items.tolist(), item_media.tolist()))
            ranked = top_media(np.array([media_of[i] for i in ids.tolist()]), vectors @ query, self._limit)
            rankings.append([m for m, _ in ranked])
            vector_rank = {m: r for r, (m, _) in enumerate(ranked, 1)}
            result.used_vectors = True
        candidates = np.unique(item_media)
        keyword = self._store.keyword_media(text, candidates, self._limit)
        keyword_rank = {m: r for r, m in enumerate(keyword, 1)}
        if keyword:
            rankings.append(keyword)
        in_text = self._store.text_media(text, candidates, self._limit)
        text_rank = {m: r for r, m in enumerate(in_text, 1)}
        if in_text:
            rankings.append(in_text)
        result.hits = self._fold_stacks([
            SearchHit(m, s, vector_rank.get(m), keyword_rank.get(m), text_rank=text_rank.get(m))
            for m, s in reciprocal_rank_fusion(rankings)])[: self._limit]
        return result

    def _fold_stacks(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Bursts of similar shots show once (the best-ranked member), with the count of the rest."""
        stacks = self._store.stack_map([h.media_id for h in hits])
        seen: set[int] = set()
        out = []
        for h in hits:
            info = stacks.get(h.media_id)
            if info is None:
                out.append(h)
            elif info[0] not in seen:
                seen.add(info[0])
                out.append(SearchHit(h.media_id, h.score, h.vector_rank, h.keyword_rank, info[2], h.text_rank))
        return out

    def similar(self, media_id: int, filters: SearchFilters = SearchFilters()) -> SearchResult:
        """Photos that look like this one."""
        query = self._store.media_vector(media_id, self._model_id)
        items, item_media = self._store.candidate_items(filters)
        result = SearchResult(candidates=len(set(item_media.tolist())))
        if query is None:
            return result
        ids, vectors = self._store.item_vectors(self._model_id, items)
        media_of = dict(zip(items.tolist(), item_media.tolist()))
        ranked = top_media(np.array([media_of[i] for i in ids.tolist()], np.int64), vectors @ query, self._limit + 1)
        result.hits = [SearchHit(m, s, r) for r, (m, s) in enumerate(ranked, 1) if m != media_id][: self._limit]
        result.used_vectors = True
        return result
