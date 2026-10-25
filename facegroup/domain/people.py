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
    threshold: float = 0.42  # default strictness: the winning votes' mean similarity must reach it
    min_votes: int = 2  # votes needed (capped at how many confirmed faces the person has)
    person_thresholds: dict[int, float] | None = None  # learned per-person strictness (overrides threshold)

    def for_person(self, person_id: int) -> float:
        return (self.person_thresholds or {}).get(person_id, self.threshold)

    @property
    def floor(self) -> float:
        """Neighbours below this never vote: the loosest threshold in use."""
        return min([self.threshold, *(self.person_thresholds or {}).values()])


@dataclass(frozen=True)
class Vote:
    person_id: int
    score: float  # mean similarity of the winning votes


def knn_assign(queries: np.ndarray, refs: np.ndarray, ref_people: np.ndarray,
               rejected: dict[int, set[int]] | None, policy: AssignPolicy,
               query_ids: list[int] | None = None, chunk: int = 2048,
               leave_one_out: bool = False, exclude: np.ndarray | None = None) -> list[Vote | None]:
    """For each query vector, the person most of its confirmed neighbours belong to, or None.

    queries: (q, d) and refs: (n, d), L2-normalised. ref_people: (n,) person id per reference face.
    rejected: {query face id: person ids this face must never be assigned to}.
    leave_one_out: queries ARE the refs; each face is matched as if it were not labelled (self-test).
    exclude: for each query, the index of its own row in refs (self-test on a sample of the refs).
    """
    if leave_one_out:
        exclude = np.arange(len(queries))
    results: list[Vote | None] = [None] * len(queries)
    if len(refs) == 0 or len(queries) == 0:
        return results
    refs_per_person = Counter(ref_people.tolist())
    k = min(policy.k, len(refs))
    rejected = rejected or {}
    floor = policy.floor
    for start in range(0, len(queries), chunk):
        sims = queries[start:start + chunk] @ refs.T
        if exclude is not None:
            rows = np.arange(len(sims))
            sims[rows, exclude[start:start + chunk]] = -np.inf  # a face may not vote for itself
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.tile(np.arange(k), (len(sims), 1))
        for row, idx in enumerate(top):
            s = sims[row, idx]
            keep = s >= floor
            if not keep.any():
                continue
            people = ref_people[idx[keep]]
            scores = s[keep]
            tally: dict[int, list[float]] = {}
            for p, sc in zip(people.tolist(), scores.tolist()):
                tally.setdefault(p, []).append(sc)
            person, votes = max(tally.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
            own = exclude is not None and ref_people[exclude[start + row]] == person
            available = refs_per_person[person] - (1 if own else 0)
            needed = max(1, min(policy.min_votes, available))
            total = int(keep.sum())
            if len(votes) < needed or len(votes) * 2 <= total:  # need a strict majority of qualifying votes
                continue
            qi = start + row
            face_id = query_ids[qi] if query_ids is not None else qi
            if person in rejected.get(face_id, ()):
                continue
            score = float(np.mean(votes))
            if score < policy.for_person(person):
                continue
            results[qi] = Vote(person, score)
    return results


def one_per_photo(votes: list[Vote | None], items: list[int], occupied: set[tuple[int, int]]) -> list[Vote | None]:
    """Two faces in one photo are two different people.

    Keeps the most confident vote for each (photo, person) and drops the rest. `occupied` holds (item, person)
    pairs already taken by faces the user confirmed, which always win.
    """
    taken = set(occupied)
    out: list[Vote | None] = [None] * len(votes)
    order = sorted((i for i, v in enumerate(votes) if v), key=lambda i: -votes[i].score)
    for i in order:
        key = (items[i], votes[i].person_id)
        if key not in taken:
            taken.add(key)
            out[i] = votes[i]
    return out


def dedupe_photos(idx: np.ndarray, items: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Within a suggested group, keep at most one face per photo: the one most like the rest of the group."""
    if len(set(items[idx].tolist())) == len(idx):
        return idx
    s = vectors[idx] @ vectors[idx].T
    np.fill_diagonal(s, 0.0)
    fit = s.sum(axis=1) / max(len(idx) - 1, 1)
    best: dict[int, int] = {}
    for pos in np.argsort(-fit):
        best.setdefault(int(items[idx[pos]]), int(idx[pos]))
    return np.array(sorted(best.values()))


@dataclass(frozen=True)
class Calibration:
    threshold: float | None  # None = not enough evidence; keep the default
    precision: float | None  # measured precision at that threshold
    samples: int  # predictions it was based on


def calibrate(scores: np.ndarray, correct: np.ndarray, target: float, min_samples: int,
              lo: float = 0.30, hi: float = 0.75, step: float = 0.01) -> Calibration:
    """Loosest threshold whose measured precision (share of matches at or above it that were right) meets
    `target`, using at least `min_samples` matches. Loosest = finds the most faces automatically."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    for t in np.arange(lo, hi + 1e-9, step):
        sel = scores >= t
        n = int(sel.sum())
        if n < min_samples:
            break  # stricter thresholds only have fewer samples
        p = float(correct[sel].mean())
        if p >= target:
            return Calibration(round(float(t), 3), round(p, 4), n)
    return Calibration(None, None, int(len(scores)))


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


def merge_groups(groups: list[np.ndarray], vectors: np.ndarray, min_similarity: float,
                 items: np.ndarray | None = None) -> list[np.ndarray]:
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
    photos = [set(items[g].tolist()) for g in groups] if items is not None else None
    while True:
        i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
        if sim[i, j] < min_similarity:
            break
        if photos is not None and photos[i] & photos[j]:
            sim[i, j] = sim[j, i] = -np.inf  # they share a photo, so they are two different people
            continue
        if photos is not None:
            photos[i] |= photos[j]
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
