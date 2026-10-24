"""People rules: who a face belongs to, and which unassigned faces look like one group. Pure computation.

See docs/ARCHITECTURE.md §5b. Only faces the user confirmed vote; automatic assignments never vote, so a
mistake cannot snowball. Rejections ("not this person") veto a vote.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssignPolicy:
    k: int = 10  # neighbours consulted
    threshold: float = 0.42  # cosine similarity a neighbour must reach to vote
    min_votes: int = 2  # votes needed (capped at how many confirmed faces the person has)


@dataclass(frozen=True)
class Vote:
    person_id: int
    score: float  # mean similarity of the winning votes


def knn_assign(queries: np.ndarray, refs: np.ndarray, ref_people: np.ndarray,
               rejected: dict[int, set[int]] | None, policy: AssignPolicy,
               query_ids: list[int] | None = None, chunk: int = 2048) -> list[Vote | None]:
    """For each query vector, the person most of its confirmed neighbours belong to, or None.

    queries: (q, d) and refs: (n, d), L2-normalised. ref_people: (n,) person id per reference face.
    rejected: {query face id: person ids this face must never be assigned to}.
    """
    results: list[Vote | None] = [None] * len(queries)
    if len(refs) == 0 or len(queries) == 0:
        return results
    refs_per_person = Counter(ref_people.tolist())
    k = min(policy.k, len(refs))
    rejected = rejected or {}
    for start in range(0, len(queries), chunk):
        sims = queries[start:start + chunk] @ refs.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.tile(np.arange(k), (len(sims), 1))
        for row, idx in enumerate(top):
            s = sims[row, idx]
            keep = s >= policy.threshold
            if not keep.any():
                continue
            people = ref_people[idx[keep]]
            scores = s[keep]
            tally: dict[int, list[float]] = {}
            for p, sc in zip(people.tolist(), scores.tolist()):
                tally.setdefault(p, []).append(sc)
            person, votes = max(tally.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
            needed = min(policy.min_votes, refs_per_person[person])
            total = int(keep.sum())
            if len(votes) < needed or len(votes) * 2 <= total:  # need a strict majority of qualifying votes
                continue
            qi = start + row
            face_id = query_ids[qi] if query_ids is not None else qi
            if person in rejected.get(face_id, ()):
                continue
            results[qi] = Vote(person, float(np.mean(votes)))
    return results


def prune_group(vectors: np.ndarray, min_similarity: float) -> np.ndarray:
    """Indices of members whose mean similarity to the *other* members is at least min_similarity.

    Drops the worst member and re-checks until all pass. Measured against the other members, not the
    centre: in a group of three, two faces of one person pull the centre towards themselves and a stranger
    still looks close to it (real-library check: centre sim 0.53, mean to others 0.11). A borderline face is
    better left unsorted than shown in the wrong group.
    """
    keep = np.arange(len(vectors))
    while len(keep) > 1:
        s = vectors[keep] @ vectors[keep].T
        np.fill_diagonal(s, 0.0)
        mean_other = s.sum(axis=1) / (len(keep) - 1)
        worst = int(np.argmin(mean_other))
        if mean_other[worst] >= min_similarity:
            break
        keep = np.delete(keep, worst)
    return keep


def merge_groups(groups: list[np.ndarray], vectors: np.ndarray, min_similarity: float) -> list[np.ndarray]:
    """Join groups of the same person that clustering split (lighting, glasses, years apart).

    Average linkage: two groups merge when their faces' mean cross-similarity reaches min_similarity,
    repeatedly, best pair first. groups: index arrays into vectors.
    """
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return groups
    # mean(A @ B.T) == (sum A) . (sum B) / (|A| |B|), so one G x G matrix gives every pair at once.
    sums = np.stack([vectors[g].sum(axis=0) for g in groups]).astype(np.float64)
    sizes = np.array([len(g) for g in groups], np.float64)
    alive = np.ones(len(groups), bool)
    sim = (sums @ sums.T) / np.outer(sizes, sizes)
    np.fill_diagonal(sim, -np.inf)
    while True:
        i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
        if sim[i, j] < min_similarity:
            break
        groups[i] = np.concatenate([groups[i], groups[j]])
        sums[i] += sums[j]
        sizes[i] += sizes[j]
        alive[j] = False
        sim[j, :] = sim[:, j] = -np.inf
        row = (sums @ sums[i]) / (sizes * sizes[i])
        row[~alive] = -np.inf
        row[i] = -np.inf
        sim[i, :] = sim[:, i] = row
    return [g for g, keep in zip(groups, alive) if keep]


def split_into_parts(vectors: np.ndarray, min_similarity: float = 0.45) -> list[np.ndarray]:
    """Break a group into parts of faces that look like the same person, biggest part first.

    Average linkage, cut where the faces of two parts average less than min_similarity to each other (the same
    cosine scale as the grouping thresholds). Within a part, faces are in similarity order, so near-duplicates sit
    together. Used to show a mixed group as "Part A / Part B" so the odd faces out are easy to pick.
    """
    n = len(vectors)
    if n <= 2:
        return [np.arange(n)] if n else []
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

    tree = linkage(vectors.astype(np.float64), method="average", metric="cosine")
    labels = fcluster(tree, t=1.0 - min_similarity, criterion="distance")
    order = leaves_list(tree)
    parts: dict[int, list[int]] = {}
    for i in order:
        parts.setdefault(int(labels[i]), []).append(int(i))
    return [np.array(p) for p in sorted(parts.values(), key=len, reverse=True)]
