"""Search rules: combining rankings and turning item scores into photo results. Pure computation."""

from __future__ import annotations

import numpy as np


def top_media(item_media: np.ndarray, scores: np.ndarray, limit: int) -> list[tuple[int, float]]:
    """Best score per photo (a video's frames collapse to its best frame), highest first."""
    if len(scores) == 0:
        return []
    order = np.argsort(-scores)
    seen: dict[int, float] = {}
    for i in order:
        m = int(item_media[i])
        if m not in seen:
            seen[m] = float(scores[i])
            if len(seen) >= limit:
                break
    return list(seen.items())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked id lists: score = sum 1 / (k + rank). Robust to the lists' scores being on
    different scales (cosine similarity vs keyword relevance), which is why it is used instead of
    adding raw scores."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, media_id in enumerate(ranking, start=1):
            fused[media_id] = fused.get(media_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


_CONNECTORS = {"and", "&", "with", ",", "+"}


def parse_people(query: str, people: list[tuple[int, str]]) -> tuple[list[int], str]:
    """Pull people's names out of a query: "Mom and Dad at the beach" -> ([mom, dad], "at the beach").

    Whole-word, case-insensitive, longest names first (so "Anna Maria" wins over "Anna"). Connecting words left
    between names are dropped. Unnamed people ("Face 12") can be found by that label too.
    """
    import re

    text = " " + query.strip() + " "
    found: list[int] = []
    for pid, name in sorted(people, key=lambda p: -len(p[1])):
        if not name.strip():
            continue
        pattern = re.compile(r"(?<![\w])" + re.escape(name.strip()) + r"(?![\w])", re.IGNORECASE)
        if pattern.search(text):
            found.append(pid)
            text = pattern.sub(" ", text)
    words = text.split()
    while words and words[0].lower() in _CONNECTORS:
        words.pop(0)
    while words and words[-1].lower() in _CONNECTORS:
        words.pop()
    cleaned = [w for i, w in enumerate(words)
               if not (w.lower() in _CONNECTORS and (i == 0 or words[i - 1].lower() in _CONNECTORS))]
    return found, " ".join(cleaned)
