"""OrganizeService: what kind of picture each photo is, which are duplicates, which form a burst.

Everything here is a label inside FaceGroup. Photo files are never deleted, moved or changed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from facegroup.domain.organize import (CATEGORY_PROMPTS, CLUTTER, PHOTO, Copy, Shot, classify, connected_sets,
                                       hash_pairs, pick_keeper, prototypes, screenshot_hint, stacks)

log = logging.getLogger(__name__)

CATEGORIES = (PHOTO, *CLUTTER)


@dataclass(frozen=True)
class OrganizePolicy:
    clutter_margin: float = 0.04  # 0 of 1,000 public photos misfiled at 0.04 (docs/EVALUATION.md §7)
    duplicate_bits: int = 6  # copies differ by <= 7 bits (99% <= 3); different photos by >= 10
    stack_gap_seconds: float = 15.0
    stack_similarity: float = 0.88  # bursts of one scene stay >= 0.90; 99.9% of different photos < 0.81


class OrganizeService:
    def __init__(self, store, embed_texts: Callable[[list[str]], np.ndarray], model_id: str,
                 thumbnail_hash: Callable[[int], int | None], policy: OrganizePolicy = OrganizePolicy()) -> None:
        self._store = store
        self._embed_texts = embed_texts
        self._model_id = model_id
        self._thumbnail_hash = thumbnail_hash
        self._p = policy
        self._protos: dict[str, np.ndarray] | None = None

    def refresh(self) -> dict[str, int]:
        """Recompute categories, duplicates and stacks. Cheap: no image is decoded except to backfill hashes."""
        rows, vectors = self._store.organize_rows(self._model_id)
        self._backfill_hashes(rows)
        cats = self._categorize(rows, vectors)
        dups = self._duplicates(rows)
        in_dup = {m for _, members, _ in dups for m in members}
        stk = self._stacks(rows, vectors, in_dup)
        log.info("organize: %d categorised, %d duplicate sets, %d stacks", cats, len(dups), len(stk))
        return {"categorised": cats, "duplicate_sets": len(dups), "stacks": len(stk)}

    # -- categories ---------------------------------------------------------------------------------------

    def _prototypes(self) -> dict[str, np.ndarray]:
        if self._protos is None:
            self._protos = prototypes({c: self._embed_texts(p) for c, p in CATEGORY_PROMPTS.items()})
        return self._protos

    def _categorize(self, rows: list[dict], vectors: np.ndarray) -> int:
        todo = [i for i, r in enumerate(rows) if r["source"] != "user"]
        if not todo:
            return 0
        with_vec = [i for i in todo if rows[i]["has_vec"]]
        results = {}
        if with_vec and self._model_id:
            for i, res in zip(with_vec, classify(vectors[with_vec], self._prototypes(), self._p.clutter_margin)):
                results[i] = (res.category, res.score)
        out = []
        for i in todo:
            r = rows[i]
            cat, score = results.get(i, (PHOTO, None))
            if screenshot_hint(r["rel_path"], r["width"], r["height"], r["taken_at"]):
                cat, score = "screenshot", 1.0
            if i in results or cat != PHOTO:
                out.append((r["id"], cat, score, self._model_id))
        self._store.write_categories(out)
        return len(out)

    def set_category(self, media_id: int, category: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(category)
        self._store.set_user_category(media_id, category)

    # -- duplicates ---------------------------------------------------------------------------------------

    def _backfill_hashes(self, rows: list[dict]) -> None:
        """Photos indexed before hashes existed: hash their cached thumbnail (same 320-px basis)."""
        missing = [r for r in rows if r["phash"] is None]
        values = []
        for r in missing:
            h = self._thumbnail_hash(r["id"])
            if h is not None:
                r["phash"] = h
                values.append((r["id"], h))
        self._store.set_phashes(values)

    def _duplicates(self, rows: list[dict]) -> list[tuple[str, list[int], int]]:
        pairs: set[tuple[int, int]] = set()
        by_hash: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            if r["content_hash"]:
                by_hash.setdefault(r["content_hash"], []).append(i)
        for members in by_hash.values():
            pairs |= {(members[0], m) for m in members[1:]}
        hashed = [i for i, r in enumerate(rows) if r["phash"] is not None]
        if hashed:
            unsigned = np.array([r["phash"] & ((1 << 64) - 1) for r in (rows[i] for i in hashed)], dtype=np.uint64)
            for a, b in hash_pairs(unsigned, self._p.duplicate_bits):
                pairs.add((hashed[a], hashed[b]))
        sets = []
        for members in connected_sets(len(rows), sorted(pairs)):
            kind = "exact" if len({rows[i]["content_hash"] for i in members}) == 1 else "near"
            copies = [Copy(rows[i]["id"], (rows[i]["width"] or 0) * (rows[i]["height"] or 0),
                           bool(rows[i]["taken_at"]), rows[i]["size"], rows[i]["rel_path"]) for i in members]
            sets.append((kind, [rows[i]["id"] for i in members], pick_keeper(copies)))
        self._store.replace_duplicates(sets)
        return sets

    # -- stacks -------------------------------------------------------------------------------------------

    def _stacks(self, rows: list[dict], vectors: np.ndarray, skip: set[int]) -> list[tuple[list[int], int]]:
        shots = []
        for i, r in enumerate(rows):
            if not r["has_vec"] or r["id"] in skip:
                continue
            shots.append(Shot(i, f"{r['root_id']}:{os.path.dirname(r['rel_path'])}", _time(r)))
        picks = self._store.stack_picks()
        out = []
        for members in stacks(shots, vectors, self._p.stack_gap_seconds, self._p.stack_similarity):
            ids = [rows[i]["id"] for i in members]
            chosen = [m for m in ids if m in picks]
            top = chosen[0] if chosen else rows[max(members, key=lambda i: (
                rows[i]["face_q"] or 0, (rows[i]["width"] or 0) * (rows[i]["height"] or 0), rows[i]["size"]))]["id"]
            out.append((ids, top))
        self._store.replace_stacks(out)
        return out

    # -- reading / per-photo actions ----------------------------------------------------------------------

    def duplicate_sets(self):
        return self._store.duplicate_sets()

    def hide_extra_copies(self, set_members: list[int], keeper: int) -> None:
        """Hide every copy except `keeper` inside FaceGroup (the files stay where they are)."""
        self._store.set_flags([m for m in set_members if m != keeper], hidden=True)
        self._store.set_flags([keeper], hidden=False)

    def summary(self) -> dict:
        sets = self._store.duplicate_sets()
        return {
            "categories": self._store.category_counts(),
            "duplicate_sets": len(sets),
            "duplicate_extra": sum(len(s["members"]) - 1 for s in sets),
            "hidden": self._store.hidden_count(),
        }


def _time(row: dict) -> float:
    if row.get("taken_at"):
        try:
            return datetime.fromisoformat(row["taken_at"]).timestamp()
        except ValueError:
            pass
    return float(row.get("mtime") or 0)
