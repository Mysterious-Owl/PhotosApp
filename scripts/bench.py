"""M1 acceptance benchmark: index N photos, kill the process mid-run, resume, and report speed and peak RAM.

    uv run python scripts/bench.py --count 10000 --kill-after 60

Builds a synthetic library of varied copies of a real group photo (flips, crops, scales, brightness,
some rotated by EXIF) under %LOCALAPPDATA%\\FaceGroup-bench, outside OneDrive, with its own data folder.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance

SAMPLE_URL = "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg"


def build_library(library: Path, count: int, seed: int = 0) -> None:
    library.mkdir(parents=True, exist_ok=True)
    existing = len(list(library.rglob("*.jpg")))
    if existing >= count:
        print(f"library: {existing} photos already present")
        return
    sample = library.parent / "t1.jpg"
    if not sample.exists():
        urllib.request.urlretrieve(SAMPLE_URL, sample)
    base = Image.open(sample).convert("RGB")
    rng = random.Random(seed)
    t0 = time.monotonic()
    for i in range(existing, count):
        img = base
        if rng.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        w, h = img.size
        cx, cy = rng.randint(0, w // 8), rng.randint(0, h // 8)
        img = img.crop((cx, cy, w - rng.randint(0, w // 8), h - rng.randint(0, h // 8)))
        scale = rng.uniform(0.8, 2.5)  # many photos larger than the source, like phone photos
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
        kwargs = {}
        if rng.random() < 0.1:  # stored sideways with an EXIF rotation tag, like many phone photos
            img = img.transpose(Image.Transpose.ROTATE_90)
            exif = Image.Exif()
            exif[0x0112] = 6
            kwargs["exif"] = exif.tobytes()
        folder = library / f"{2015 + i % 10}" / f"album-{i % 37:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        img.save(folder / f"IMG_{i:06d}.jpg", quality=88, **kwargs)
        if (i + 1) % 500 == 0:
            print(f"  generated {i + 1}/{count} ({time.monotonic() - t0:.0f}s)")


def run_scan(library: Path, env: dict, kill_after: float | None) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "facegroup.entrypoints.cli", "scan", str(library)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    if kill_after is None:
        out, _ = proc.communicate()
        return proc.returncode, out
    try:
        out, _ = proc.communicate(timeout=kill_after)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        proc.kill()  # hard kill: no cleanup, like a crash or power loss
        out, _ = proc.communicate()
        return -9, out


def job_counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(db)
    try:
        return {f"{s}.{st}": n for s, st, n in conn.execute("SELECT stage, state, COUNT(*) FROM job GROUP BY 1, 2")}
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--kill-after", type=float, default=None, help="seconds before hard-killing the first run")
    ap.add_argument("--base", type=Path, default=Path(os.environ["LOCALAPPDATA"]) / "FaceGroup-bench")
    ap.add_argument("--fresh", action="store_true", help="delete the bench database first")
    args = ap.parse_args()

    library, data = args.base / "library", args.base / "data"
    build_library(library, args.count)
    if args.fresh and data.exists():
        for name in ("facegroup.db", "facegroup.db-wal", "facegroup.db-shm"):
            (data / name).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(data / "cache", ignore_errors=True)
    data.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "FACEGROUP_DATA_DIR": str(data), "PYTHONIOENCODING": "utf-8"}
    # Reuse the already-downloaded model instead of fetching it again.
    models = Path(os.environ["LOCALAPPDATA"]) / "FaceGroup" / "models"
    if models.exists() and not (data / "models").exists():
        os.symlink(models, data / "models", target_is_directory=True) if os.name != "nt" else \
            subprocess.run(["cmd", "/c", "mklink", "/J", str(data / "models"), str(models)], capture_output=True)

    t0 = time.monotonic()
    code, out = run_scan(library, env, args.kill_after)
    print(out.strip().splitlines()[-1] if out.strip() else "", f"\nrun 1 exit={code} after {time.monotonic() - t0:.0f}s")
    if code == -9:
        print("killed mid-run; job states:", job_counts(data / "facegroup.db"))
        t1 = time.monotonic()
        code, out = run_scan(library, env, None)
        print(out)
        print(f"run 2 (resume) exit={code} after {time.monotonic() - t1:.0f}s")
    else:
        print(out)
    print(f"total wall time {time.monotonic() - t0:.0f}s")
    print("final job states:", job_counts(data / "facegroup.db"))


if __name__ == "__main__":
    main()
