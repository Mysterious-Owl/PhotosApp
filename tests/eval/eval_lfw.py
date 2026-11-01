"""LFW face verification with FaceGroup's own pipeline (decoder -> SCRFD port -> alignment -> ArcFace).

    uv run python tests/eval/eval_lfw.py [--device auto|cpu|directml]

Checks that our re-implementation of InsightFace's detection and alignment preserves the model's published
accuracy (buffalo_l: LFW 99.83%, InsightFace model zoo). A silent alignment bug would show up here as a
lower score long before it showed up anywhere else.

Standard protocol: 6,000 pairs (3,000 same person, 3,000 different) in 10 folds; the threshold is chosen on
9 folds and accuracy measured on the held-out fold. Data (~173 MB, research use) is downloaded once to
data\\eval\\lfw (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from facegroup.adapters.media.decoder import PillowDecoder
from facegroup.adapters.models.buffalo_onnx import ARCFACE_DST, BuffaloL, ensure_buffalo_l, norm_crop
from facegroup.config import default_data_dir, default_eval_dir

LFW_URL = "https://ndownloader.figshare.com/files/5976018"  # lfw.tgz, original 250x250 images
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"  # pairs.txt
PUBLISHED_LFW = 99.83


def fetch(base: Path) -> tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    pairs = base / "pairs.txt"
    if not pairs.exists():
        urllib.request.urlretrieve(PAIRS_URL, pairs)
    images = base / "lfw"
    if not images.exists():
        archive = base / "lfw.tgz"
        if not archive.exists():
            print("downloading LFW (~173 MB)...")
            urllib.request.urlretrieve(LFW_URL, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(base, filter="data")  # untrusted archive: refuse absolute paths, links, devices
    return pairs, images


def read_pairs(path: Path) -> tuple[list[tuple[str, str, bool]], int]:
    lines = path.read_text().strip().splitlines()
    folds, per = (int(x) for x in lines[0].split())
    pairs = []
    for line in lines[1:]:
        p = line.split()
        if len(p) == 3:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[0]}/{p[0]}_{int(p[2]):04d}.jpg", True))
        else:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[2]}/{p[2]}_{int(p[3]):04d}.jpg", False))
    assert len(pairs) == folds * per * 2, len(pairs)
    return pairs, folds


def embed_all(model: BuffaloL, images: Path, names: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Embed the most central face of each image. Images with no detection fall back to a fixed centre
    alignment (LFW faces are centred), and are listed so they can be reported."""
    decoder = PillowDecoder()
    fallback_lm = ARCFACE_DST * 1.25 + np.array([125 - 56 * 1.25, 125 - 64 * 1.25], dtype=np.float32)
    out, missed = {}, []
    t0 = time.monotonic()
    for i, name in enumerate(names):
        rgb = decoder.decode(images / name).rgb
        faces = model.analyze(rgb)
        h, w = rgb.shape[:2]
        if faces:
            def dist(f):
                x1, y1, x2, y2 = f.bbox
                return ((x1 + x2) / 2 - w / 2) ** 2 + ((y1 + y2) / 2 - h / 2) ** 2
            out[name] = min(faces, key=dist).embedding
        else:
            missed.append(name)
            out[name] = model._embed([norm_crop(rgb, fallback_lm)])[0]
        if (i + 1) % 1000 == 0:
            print(f"  embedded {i + 1}/{len(names)} ({(i + 1) / (time.monotonic() - t0):.1f} img/s)")
    return out, missed


def kfold_accuracy(sims: np.ndarray, same: np.ndarray, folds: int) -> tuple[float, float, list[float]]:
    idx = np.arange(len(sims))
    fold_of = idx // (len(sims) // folds)
    candidates = np.linspace(-1, 1, 4001)
    accs, thresholds = [], []
    for k in range(folds):
        train, test = fold_of != k, fold_of == k
        train_acc = [((sims[train] >= t) == same[train]).mean() for t in candidates]
        t = candidates[int(np.argmax(train_acc))]
        thresholds.append(float(t))
        accs.append(((sims[test] >= t) == same[test]).mean())
    return float(np.mean(accs)) * 100, float(np.std(accs)) * 100, thresholds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--base", type=Path, default=default_eval_dir() / "lfw")
    args = ap.parse_args()

    pairs_file, images = fetch(args.base)
    pairs, folds = read_pairs(pairs_file)
    names = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    model = BuffaloL(ensure_buffalo_l(default_data_dir() / "models"), device=args.device)
    print(f"{len(pairs)} pairs, {len(names)} images, provider {model.provider}")
    emb, missed = embed_all(model, images, names)

    sims = np.array([float(emb[a] @ emb[b]) for a, b, _ in pairs])
    same = np.array([s for _, _, s in pairs])
    acc, std, thresholds = kfold_accuracy(sims, same, folds)
    # Errors at the chosen threshold, kept for inspection.
    t = float(np.median(thresholds))
    wrong = [(a, b, s, round(float(x), 3)) for (a, b, s), x in zip(pairs, sims) if (x >= t) != s]
    result = {
        "provider": model.provider,
        "pairs": len(pairs),
        "images": len(names),
        "no_face_detected": len(missed),
        "no_face_examples": missed[:20],
        "accuracy_pct": round(acc, 3),
        "accuracy_std_pct": round(std, 3),
        "published_pct": PUBLISHED_LFW,
        "threshold_median": round(t, 4),
        "threshold_range": [round(min(thresholds), 4), round(max(thresholds), 4)],
        "same_person_similarity_p1_p5_p50": [round(float(np.percentile(sims[same], q)), 3) for q in (1, 5, 50)],
        "different_people_similarity_p50_p95_p99": [round(float(np.percentile(sims[~same], q)), 3) for q in (50, 95, 99)],
        "errors": wrong,
    }
    out = args.base.parent / f"lfw_result_{model.provider}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=2))
    print(f"{len(wrong)} misclassified pairs at threshold {t:.3f}; details in {out}")


if __name__ == "__main__":
    main()
