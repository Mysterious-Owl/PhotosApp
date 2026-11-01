"""Organising rules: what kind of picture it is, which pictures are duplicates, which form a burst. Pure computation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# -- categories --------------------------------------------------------------------------------------------

PHOTO = "photo"
CLUTTER = ("screenshot", "document", "receipt", "meme", "illustration")

# Short descriptions per category, matched against the image embedding (zero-shot: no new model needed).
CATEGORY_PROMPTS: dict[str, list[str]] = {
    PHOTO: [
        "a photo of people", "a photo of a person", "a group photo of friends", "a photo of a landscape",
        "a photo of food on a plate", "a photo of a city street", "a photo of an animal", "a photo of a room",
        "a selfie", "a photo taken with a phone camera",
    ],
    "screenshot": [
        "a screenshot of a phone screen", "a screenshot of a website", "a screenshot of a computer desktop",
        "a screenshot of a chat conversation", "a screenshot of an app", "a screenshot of a video call",
    ],
    "document": [
        "a scanned document page with text", "a page of a book", "a photo of a printed document",
        "a slide from a presentation", "handwritten notes on paper",
    ],
    "receipt": ["a photo of a receipt", "a shopping bill with prices", "an invoice with a total amount"],
    "meme": ["a meme with text on a picture", "an internet meme", "a comic strip with speech bubbles"],
    "illustration": [
        "a digital illustration", "an anime wallpaper", "a cartoon drawing", "a computer generated wallpaper",
        "abstract digital art",
    ],
}

# Matched against the file name only. "snip" and "capture" must be whole words ("Capture.PNG", "Snip 3.png"),
# so names like "Captured" or "Snippets" don't count.
_SCREENSHOT_NAME = re.compile(
    r"(screen\s*shot|screenshot|scrnshot|screen[_ ]?recording|(?<![a-z])(?:snip|capture)(?![a-z]))", re.I)
_SCREEN_WIDTHS = {720, 750, 828, 1080, 1125, 1170, 1179, 1242, 1284, 1290, 1320, 1440, 1536, 1600, 1920, 2048,
                  2160, 2560, 2736, 2880, 3024, 3840}


def screenshot_hint(rel_path: str, width: int | None, height: int | None, taken_at: str | None) -> bool:
    """Strong, cheap evidence: a 'Screenshot' file name, or a PNG with no camera date at an exact screen size."""
    name = rel_path.replace("\\", "/").rsplit("/", 1)[-1]
    if _SCREENSHOT_NAME.search(name):
        return True
    if name.lower().endswith(".png") and not taken_at and width and height:
        return min(width, height) in _SCREEN_WIDTHS or max(width, height) in _SCREEN_WIDTHS
    return False


@dataclass(frozen=True)
class CategoryResult:
    category: str
    score: float  # margin of the winning category over the best "photo" description


def prototypes(prompt_vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each category's descriptions averaged into one unit vector."""
    out = {}
    for cat, vecs in prompt_vectors.items():
        v = vecs.mean(axis=0)
        out[cat] = v / np.linalg.norm(v)
    return out


def classify(image_vectors: np.ndarray, protos: dict[str, np.ndarray], margin: float) -> list[CategoryResult]:
    """Clutter only when a clutter category beats "photo" by `margin`: a real photo hidden by mistake is worse
    than a screenshot left visible."""
    cats = list(protos)
    sims = image_vectors @ np.stack([protos[c] for c in cats]).T
    photo = sims[:, cats.index(PHOTO)]
    out = []
    for row, p in zip(sims, photo):
        best = int(np.argmax(np.where(np.array(cats) == PHOTO, -np.inf, row)))
        lead = float(row[best] - p)
        out.append(CategoryResult(cats[best], round(lead, 4)) if lead >= margin else CategoryResult(PHOTO, round(lead, 4)))
    return out


# -- duplicates --------------------------------------------------------------------------------------------

def hash_pairs(hashes: np.ndarray, max_distance: int = 6) -> list[tuple[int, int]]:
    """Index pairs whose 64-bit difference hashes differ in at most `max_distance` bits.

    Measured on public photos (docs/EVALUATION.md §7): resized/recompressed copies differ by <= 7 bits (99% <= 3),
    different photos by >= 10, so 6 separates them. Avoids comparing every pair: the hash is cut into
    max_distance + 1 bands, and two hashes within max_distance bits must agree exactly on at least one band.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    # Flat images (blank screenshots, black frames) hash to nearly all-0 or all-1 bits and would all "match";
    # real photos sit near 32 set bits. Leave degenerate hashes out (exact copies are still found by file hash).
    bits = np.unpackbits(h.view(np.uint8).reshape(n, 8), axis=1).sum(axis=1) if n else np.zeros(0)
    usable = (bits >= 8) & (bits <= 56)
    bands = max_distance + 1
    edges = np.linspace(0, 64, bands + 1).astype(int)
    seen: set[tuple[int, int]] = set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = np.uint64(((1 << (hi - lo)) - 1) << lo)
        keys = (h & mask) >> np.uint64(lo)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            members = order[s:e]
            members = members[usable[members]]
            if len(members) < 2:
                continue
            x = h[members][:, None] ^ h[members][None, :]
            dist = np.unpackbits(x.view(np.uint8).reshape(len(members), len(members), 8), axis=2).sum(axis=2)
            rows, cols = np.nonzero(np.triu(dist <= max_distance, k=1))
            for r, c in zip(rows.tolist(), cols.tolist()):
                a, b = int(members[r]), int(members[c])
                seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def near_pairs(vectors: np.ndarray, threshold: float, chunk: int = 2048) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose embeddings are at least `threshold` similar."""
    pairs: list[tuple[int, int]] = []
    n = len(vectors)
    for start in range(0, n, chunk):
        block = vectors[start:start + chunk] @ vectors.T
        rows, cols = np.nonzero(block >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < c:
                pairs.append((i, c))
    return pairs


def connected_sets(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find: groups of indices linked by pairs (sets of 2+ only)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    sets: dict[int, list[int]] = {}
    for i in range(n):
        sets.setdefault(find(i), []).append(i)
    return [s for s in sets.values() if len(s) > 1]


@dataclass(frozen=True)
class Copy:
    media_id: int
    pixels: int
    has_date: bool
    size: int
    path: str


def pick_keeper(copies: list[Copy]) -> int:
    """The copy worth keeping: most pixels, then a capture date, then the larger file, then the shorter path."""
    return max(copies, key=lambda c: (c.pixels, c.has_date, c.size, -len(c.path))).media_id


# -- stacks (bursts) ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shot:
    index: int  # into the embedding matrix
    folder: str
    time: float  # seconds since epoch (capture time, else file time)


def stacks(shots: list[Shot], vectors: np.ndarray, max_gap: float = 15.0, min_similarity: float = 0.88) -> list[list[int]]:
    """Runs of similar shots taken within `max_gap` seconds of each other in the same folder (2+ shots)."""
    out: list[list[int]] = []
    ordered = sorted(shots, key=lambda s: (s.folder, s.time))
    run: list[Shot] = []
    for shot in ordered:
        if run and shot.folder == run[-1].folder and shot.time - run[-1].time <= max_gap \
                and float(vectors[shot.index] @ vectors[run[-1].index]) >= min_similarity:
            run.append(shot)
            continue
        if len(run) > 1:
            out.append([s.index for s in run])
        run = [shot]
    if len(run) > 1:
        out.append([s.index for s in run])
    return out
