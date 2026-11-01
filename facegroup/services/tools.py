"""File tools: Convert HEIC, Remove metadata, Duplicate files. They work on any folder the user picks.

Convert and Remove metadata only read the originals and write new files into a new folder. Duplicate files is the
one feature that changes the user's folders: after review and confirmation, extra copies move into a holding folder
on the same drive, every move is logged, and Restore puts them back. Nothing is ever deleted.

Jobs run one at a time on a background thread (ToolRunner), separate from photo indexing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from facegroup.adapters.tools import images, mover, walk
from facegroup.domain.files import (
    CONVERTED_PREFIX,
    HEIC_EXTENSIONS,
    HOLDING_FOLDER,
    METADATA_EXTENSIONS,
    PHOTO_VIDEO_EXTENSIONS,
    Copy,
    Decision,
    converted_candidates,
    converted_name,
    decide,
    folder_key,
    folder_pairs,
    holding_path,
    is_converted_set,
    pair_key,
)

log = logging.getLogger(__name__)


class Busy(RuntimeError):
    pass


@dataclass
class Job:
    """What the UI shows about the current or last tool job."""

    kind: str = ""
    title: str = ""
    state: str = "idle"  # idle | running | done | stopped | error
    done: int = 0
    total: int = 0
    message: str = ""
    result: dict = field(default_factory=dict)
    started: float = 0.0
    finished: float = 0.0

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def percent(self) -> int:
        return int(100 * self.done / self.total) if self.total else 0


class ToolRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.job = Job()

    def start(self, kind: str, title: str, fn: Callable[[ToolRunner], dict]) -> None:
        with self._lock:
            if self.job.running:
                raise Busy(f"{self.job.title} is still running.")
            self._stop.clear()
            self.job = Job(kind, title, "running", message="Starting…", started=time.time())
        threading.Thread(target=self._run, args=(fn,), name=f"tool-{kind}", daemon=True).start()

    def _run(self, fn) -> None:
        job = self.job
        try:
            job.result = fn(self) or {}
            job.state = "stopped" if self._stop.is_set() else "done"
            job.message = "Stopped." if self._stop.is_set() else job.result.get("summary", "Done.")
        except Exception as exc:  # shown to the user; the details go to the log
            log.exception("tool job %s failed", job.kind)
            job.state, job.message = "error", f"Failed: {exc}"
        job.finished = time.time()

    def stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def progress(self, done: int, total: int, message: str = "") -> None:
        self.job.done, self.job.total = done, total
        if message:
            self.job.message = message

    def wait(self, timeout: float = 30.0) -> Job:
        """Tests and the CLI: block until the job finishes."""
        end = time.time() + timeout
        while self.job.running and time.time() < end:
            time.sleep(0.02)
        return self.job


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _files(folder: Path, extensions: frozenset[str] | None, recurse: bool, runner: ToolRunner) -> list[Path]:
    if recurse:
        return [Path(f.path) for f in walk.walk([folder], extensions, runner.should_stop) if not f.online_only]
    out = []
    for entry in os.scandir(folder):
        if entry.is_file() and (extensions is None or Path(entry.name).suffix.lower() in extensions):
            if not walk.is_online_only(entry.stat()):
                out.append(Path(entry.path))
    return sorted(out)


def _parallel(paths: list[Path], work: Callable[[Path], str], runner: ToolRunner, verb: str) -> dict:
    """Run `work` on every path with one worker per CPU; counts its return values. Stop finishes files in hand."""
    counts: dict[str, int] = defaultdict(int)
    errors: list[str] = []
    workers = os.cpu_count() or 4
    total = len(paths)
    runner.progress(0, total, f"{verb} {total} files")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        it = iter(paths)
        for p in it:  # keep a bounded number in flight, so Stop is quick
            futures[pool.submit(work, p)] = p
            if len(futures) >= workers * 2:
                break
        done = 0
        while futures:
            fut = next(iter(futures))
            p = futures.pop(fut)
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                errors.append(f"{p.name}: {exc}")
            done += 1
            runner.progress(done, total)
            if not runner.should_stop():
                nxt = next(it, None)
                if nxt is not None:
                    futures[pool.submit(work, nxt)] = nxt
    return {"counts": dict(counts), "errors": errors[:200]}


# -- Convert HEIC ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvertOptions:
    folder: Path
    output: Path
    fmt: str = "png"  # png | jpg
    copy_others: bool = False
    overwrite: bool = False
    recurse: bool = False


def run_convert(opts: ConvertOptions, runner: ToolRunner) -> dict:
    exts = None if opts.copy_others else HEIC_EXTENSIONS
    paths = [p for p in _files(opts.folder, exts, opts.recurse, runner) if opts.output not in p.parents]

    def work(src: Path) -> str:
        rel = src.relative_to(opts.folder)
        is_heic = src.suffix.lower() in HEIC_EXTENSIONS
        dst = opts.output / rel.parent / converted_name(src.name, opts.fmt)
        if dst.exists() and not opts.overwrite:
            return "skipped"
        if is_heic:
            images.convert_heic(src, dst, opts.fmt)
            return "converted"
        images.copy_file(src, dst)
        return "copied"

    out = _parallel(paths, work, runner, "Converting")
    c = out["counts"]
    out["output"] = str(opts.output)
    out["summary"] = (f"Converted {c.get('converted', 0)}, copied {c.get('copied', 0)}, "
                      f"skipped {c.get('skipped', 0)} already there, {c.get('failed', 0)} failed.")
    return out


# -- Remove metadata --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MetadataOptions:
    folder: Path
    output: Path
    keep_icc: bool = True
    keep_date: bool = False
    overwrite: bool = False
    recurse: bool = True


def run_metadata_check(folder: Path, recurse: bool, runner: ToolRunner) -> dict:
    """Counts of what metadata the pictures carry, plus which files have a location. Nothing is written."""
    paths = _files(folder, METADATA_EXTENSIONS, recurse, runner)
    with_gps: list[str] = []
    tally: dict[str, int] = defaultdict(int)

    def work(p: Path) -> str:
        r = images.inspect(p)
        for key, present in (("gps", r.gps), ("date", r.date_taken), ("exif", r.exif_tags > 0), ("xmp", r.xmp),
                             ("iptc", r.iptc), ("comments", r.comments > 0), ("clean", r.clean)):
            if present:
                tally[key] += 1  # each key is only touched by +=, safe enough under the GIL for counts
        if r.gps:
            with_gps.append(str(p.relative_to(folder)))
        return "checked"

    out = _parallel(paths, work, runner, "Checking")
    out.update(tally=dict(tally), files=len(paths), with_gps=sorted(with_gps)[:500])
    out["summary"] = (f"{len(paths)} pictures: {tally.get('gps', 0)} with a location, "
                      f"{tally.get('exif', 0)} with camera data, {tally.get('clean', 0)} already clean.")
    return out


def run_metadata_clean(opts: MetadataOptions, runner: ToolRunner) -> dict:
    paths = [p for p in _files(opts.folder, METADATA_EXTENSIONS, opts.recurse, runner) if opts.output not in p.parents]

    def work(src: Path) -> str:
        dst = opts.output / src.relative_to(opts.folder)
        if dst.exists() and not opts.overwrite:
            return "skipped"
        images.strip_metadata(src, dst, keep_icc=opts.keep_icc, keep_date=opts.keep_date)
        return "cleaned"

    out = _parallel(paths, work, runner, "Cleaning")
    c = out["counts"]
    out["output"] = str(opts.output)
    out["summary"] = (f"Cleaned {c.get('cleaned', 0)}, skipped {c.get('skipped', 0)} already there, "
                      f"{c.get('failed', 0)} failed.")
    return out


# -- Duplicate files -------------------------------------------------------------------------------------------

class DuplicatesService:
    def __init__(self, store) -> None:
        self._store = store
        self._empty: list[str] = []  # empty folders from the last scan (cheap to recompute, so not stored)

    # scanning

    def run_scan(self, roots: list[Path], media_only: bool, min_size: int, runner: ToolRunner) -> dict:
        """Size first (free), then a 128 KB quick hash, then a full hash: only files that survive each step are read
        further. Online-only files are counted but never read."""
        started = _now()
        errors: list[str] = []
        by_size: dict[int, list[walk.FoundFile]] = defaultdict(list)
        heics: list[Copy] = []
        pngs: list[Copy] = []
        files = total_bytes = cloud = 0
        runner.progress(0, 0, "Listing files…")
        for f in walk.walk(roots, PHOTO_VIDEO_EXTENSIONS if media_only else None, runner.should_stop, errors):
            if f.size < min_size:
                continue
            files += 1
            total_bytes += f.size
            if f.online_only:
                cloud += 1
                continue
            by_size[f.size].append(f)
            ext = Path(f.path).suffix.lower()
            if ext in HEIC_EXTENSIONS:
                heics.append(Copy(f.path, f.size, f.mtime))
            elif ext == ".png":
                pngs.append(Copy(f.path, f.size, f.mtime))
            if files % 2000 == 0:
                runner.progress(0, 0, f"Listing files… {files:,} so far")
        candidates = [group for group in by_size.values() if len(group) > 1]
        groups = self._hash_groups(candidates, runner, errors)
        converted = self._converted_groups(heics, pngs, groups, runner, errors)
        self._empty = walk.empty_folders(roots, runner.should_stop)
        if runner.should_stop():
            return {"summary": "Stopped: the previous results are kept."}
        self._store.save_scan([str(r) for r in roots], started, groups + converted, files, total_bytes, cloud, errors)
        extra = sum(size * (len(copies) - 1) for _, size, copies in groups)
        heic = f"{len(converted):,} HEIC photos that also have a PNG copy. " if converted else ""
        return {"summary": f"{len(groups):,} sets of identical files; {_gb(extra)} in extra copies. {heic}"
                           f"{len(self._empty):,} empty folders."}

    def _converted_groups(self, heics: list[Copy], pngs: list[Copy], identical, runner: ToolRunner,
                          errors: list[str]) -> list[tuple[str, int, list[Copy]]]:
        """HEIC photos with a PNG of the same name that holds the same picture (checked by pixels, so a different
        photo that happens to share a name like IMG_0001 is not matched). Files already in an identical set are
        left for that set; the next scan pairs whichever copy remains."""
        in_sets = {folder_key(c.path) for _, _, copies in identical for c in copies}
        pairs = converted_candidates([h for h in heics if folder_key(h.path) not in in_sets],
                                     [p for p in pngs if folder_key(p.path) not in in_sets])
        if not pairs:
            return []
        runner.progress(0, len(pairs), "Matching HEIC photos with their PNG copies")

        def same(hp: tuple[Copy, Copy]) -> bool:
            try:
                return images.same_picture(Path(hp[0].path), Path(hp[1].path))
            except Exception as exc:  # an unreadable or corrupt picture is simply not a match
                errors.append(f"{hp[0].path}: {exc}")
                return False

        used: set[str] = set()
        out = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            checks = pool.map(same, pairs)
            for i, ((h, p), same) in enumerate(zip(pairs, checks)):
                runner.progress(i + 1, len(pairs))
                if runner.should_stop():
                    break
                if same and folder_key(h.path) not in used and folder_key(p.path) not in used:
                    used |= {folder_key(h.path), folder_key(p.path)}
                    digest = _try(walk.quick_hash, p.path, errors)
                    if digest:
                        out.append((CONVERTED_PREFIX + digest, p.size, [h, p]))
        return out

    def _hash_groups(self, candidates, runner: ToolRunner, errors: list[str]) -> list[tuple[str, int, list[Copy]]]:
        total = sum(len(g) for g in candidates)
        done = 0
        groups: list[tuple[str, int, list[Copy]]] = []
        with ThreadPoolExecutor(max_workers=4) as pool:  # disk-bound: a few readers is enough
            for same_size in candidates:
                if runner.should_stop():
                    break
                for step, fn in (("quick", walk.quick_hash), ("full", walk.full_hash)):
                    digests = list(pool.map(lambda f, fn=fn: _try(fn, f.path, errors), same_size))
                    buckets: dict[str, list[walk.FoundFile]] = defaultdict(list)
                    for f, d in zip(same_size, digests):
                        if d is not None:
                            buckets[d].append(f)
                    if step == "quick":
                        same_size = [f for b in buckets.values() if len(b) > 1 for f in b]
                        if not same_size:
                            break
                    else:
                        for d, fs in buckets.items():
                            if len(fs) > 1:
                                groups.append((d, fs[0].size, [Copy(f.path, f.size, f.mtime) for f in fs]))
                done += len(same_size) or 1
                runner.progress(min(done, total), total, "Comparing files with the same size")
        return groups

    # reviewing

    def summary(self) -> dict | None:
        scan = self._store.latest_scan()
        if scan is None:
            return None
        decisions = self.decisions()
        by_reason: dict[str, list[Decision]] = defaultdict(list)
        for _, _, d in decisions:
            by_reason[d.reason].append(d)

        def extra(ds):
            return sum(c.size for d in ds for c in d.remove)

        ready = [d for _, _, d in decisions if d.confirmed]
        suggested = by_reason.get("suggested", [])
        converted = by_reason.get("converted", [])
        return {"scan": scan, "groups": len(decisions), "ready": len(ready), "ready_files": sum(len(d.remove) for d in ready),
                "ready_bytes": extra(ready), "suggested": len(suggested),
                "suggested_files": sum(len(d.remove) for d in suggested), "suggested_bytes": extra(suggested),
                "converted": len(converted), "converted_bytes": extra(converted),
                "by_reason": {k: len(v) for k, v in by_reason.items()}, "empty": len(self._empty)}

    def decisions(self) -> list[tuple[int, str, Decision]]:
        prefs, choices = self._store.folder_prefs(), self._store.dup_choices()
        return [(gid, digest, decide(copies, prefs, choices.get(digest), converted=is_converted_set(digest)))
                for gid, digest, copies in self._store.dup_groups()]

    def pairs(self, undecided_only: bool = False):
        # HEIC + PNG sets are decided by format, not by folder.
        groups = [copies for _, digest, copies in self._store.dup_groups() if not is_converted_set(digest)]
        pairs = folder_pairs(groups, self._store.folder_prefs())
        return [p for p in pairs if p.winner is None] if undecided_only else pairs

    def prefer(self, folder_a: str, folder_b: str, keep: str | None) -> None:
        """Between these two folders, keep the copy in `keep` (or forget the preference with None)."""
        if keep is not None and folder_key(keep) not in (folder_key(folder_a), folder_key(folder_b)):
            raise ValueError("keep must be one of the two folders")
        self._store.set_folder_pref(pair_key(folder_a, folder_b), folder_key(keep) if keep else None)

    def choose(self, digest: str, keep_path: str | None) -> None:
        self._store.set_dup_choice(digest, keep_path)

    def empty_folders(self) -> list[str]:
        return list(self._empty)

    # acting

    def run_remove(self, include_suggested: bool, runner: ToolRunner, include_converted: bool = False) -> dict:
        """Move every extra copy of a decided group into the holding folder on its drive. Suggestions, and HEIC
        originals that have a PNG copy, only when asked. Before each move the kept copy and the copy being moved are
        checked again (still there, same size and date), so a file that changed since the scan is left alone."""
        todo = [(digest, d) for _, digest, d in self.decisions()
                if d.confirmed or (include_suggested and d.reason == "suggested")
                or (include_converted and d.reason == "converted")]
        total = sum(len(d.remove) for _, d in todo)
        batch = self._store.start_batch("duplicates")
        taken = self._store.holding_paths()
        moved, skipped, freed = [], [], 0
        runner.progress(0, total, f"Moving {total} extra copies")
        for _, d in todo:
            if runner.should_stop():
                break
            if not _unchanged(d.keep):
                skipped.append(f"{d.keep.path}: the copy to keep changed or is gone")
                continue
            for c in d.remove:
                if runner.should_stop():
                    break
                if not _unchanged(c):
                    skipped.append(f"{c.path}: changed since the scan")
                    continue
                target = holding_path(c.path, taken)
                try:
                    mover.move_file(c.path, target)
                except OSError as exc:
                    skipped.append(f"{c.path}: {exc}")
                    continue
                taken.add(target.casefold())
                self._store.record_removed(batch, c.path, target, c.size, d.keep.path)
                moved.append(c.path)
                freed += c.size
                runner.progress(len(moved) + len(skipped), total)
        self._store.forget_paths(moved)
        return {"summary": f"Moved {len(moved):,} extra copies ({_gb(freed)}) to “{HOLDING_FOLDER}” folders. "
                           f"Restore puts them back.", "skipped": skipped[:200], "batch": batch}

    def remove_empty_folders(self, folders: list[str]) -> int:
        """Remove the given folders if they are still empty (deepest first). Logged, so Restore recreates them."""
        known = {folder_key(f) for f in self._empty}
        batch = self._store.start_batch("empty_folders")
        n = 0
        for f in sorted(folders, key=len, reverse=True):
            if folder_key(f) in known and mover.remove_empty_folder(f):
                self._store.record_removed(batch, f, None, 0, None)
                n += 1
        self._empty = [f for f in self._empty if os.path.isdir(f)]
        return n

    def batches(self) -> list[dict]:
        return self._store.batches()

    def batch_files(self, batch_id: int) -> list[dict]:
        return self._store.batch_files(batch_id)

    def run_restore(self, batch_id: int, runner: ToolRunner) -> dict:
        rows = [r for r in self._store.batch_files(batch_id) if r["restored_at"] is None]
        restored, problems = 0, []
        for i, r in enumerate(reversed(rows)):  # folders: shallowest first
            if runner.should_stop():
                break
            try:
                if r["holding"] is None:
                    os.makedirs(r["original"], exist_ok=True)
                else:
                    mover.move_file(r["holding"], r["original"])
                    mover.prune_empty_parents(r["holding"], HOLDING_FOLDER)
                self._store.mark_restored(r["id"])
                restored += 1
            except OSError as exc:
                problems.append(f"{r['original']}: {exc}")
            runner.progress(i + 1, len(rows))
        return {"summary": f"Restored {restored:,} of {len(rows):,}.", "skipped": problems[:200]}


def _try(fn, path: str, errors: list[str]) -> str | None:
    try:
        return fn(path)
    except OSError as exc:
        errors.append(f"{path}: {exc.strerror or exc}")
        return None


def _unchanged(c: Copy) -> bool:
    try:
        st = os.stat(c.path)
    except OSError:
        return False
    return st.st_size == c.size and abs(st.st_mtime - c.mtime) < 2


def _gb(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n / 1024:.0f} KB"
