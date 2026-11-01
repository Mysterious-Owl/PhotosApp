"""Check the app's ONNX search embedder against the reference implementation, on Flickr30k 1K.

    uv run python tests/eval/eval_retrieval_onnx.py [--variant quantized|fp16|""] [--device auto|cpu]

Run tests/eval/eval_retrieval.py first (it downloads the data and saves reference embeddings). Reports:
retrieval recall (same protocol), agreement with the reference vectors (cosine), and speed per image.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from facegroup.adapters.models.siglip2_onnx import SigLip2
from facegroup.config import default_data_dir, default_eval_dir

BASE = default_eval_dir()


def load_flickr():
    import ast
    import csv

    with open(BASE / "flickr1k" / "test_1k_flickr.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    images, captions, owner = [], [], []
    for i, row in enumerate(rows):
        images.append(BASE / "flickr1k" / "images" / Path(row["filename"]).name)
        for c in ast.literal_eval(row["raw"])[:5]:
            captions.append(c)
            owner.append(i)
    return images, captions, np.array(owner)


def recall(sim, owner):
    out = {}
    r = (-sim).argsort(axis=1)
    for k in (1, 5, 10):
        out[f"t2i_R@{k}"] = round(float((r[:, :k] == owner[:, None]).any(axis=1).mean() * 100), 2)
    r = (-sim.T).argsort(axis=1)
    for k in (1, 5, 10):
        out[f"i2t_R@{k}"] = round(float(np.mean([(owner[r[i, :k]] == i).any() for i in range(sim.shape[1])]) * 100), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fp16", help="fp16, or an empty string for fp32")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    images, captions, owner = load_flickr()
    model = SigLip2(default_data_dir() / "models", device=args.device, variant=args.variant)
    model.embed_images([np.asarray(Image.open(images[0]).convert("RGB"))])  # warm-up
    t0 = time.monotonic()
    ie = np.concatenate([model.embed_images([np.asarray(Image.open(p).convert("RGB")) for p in images[i:i + 16]])
                         for i in range(0, len(images), 16)])
    per_image = (time.monotonic() - t0) / len(images)
    t1 = time.monotonic()
    te = np.concatenate([model.embed_queries(captions[i:i + 128]) for i in range(0, len(captions), 128)])
    per_caption = (time.monotonic() - t1) / len(captions)
    result = recall(te @ ie.T, owner)
    ref_i = BASE / "ref_siglip2-base_images.npy"
    if ref_i.exists():
        ri, rt = np.load(ref_i), np.load(BASE / "ref_siglip2-base_texts.npy")
        result["image_cosine_to_reference_min_mean"] = [round(float((ri * ie).sum(1).min()), 4),
                                                        round(float((ri * ie).sum(1).mean()), 4)]
        result["text_cosine_to_reference_min_mean"] = [round(float((rt * te).sum(1).min()), 4),
                                                       round(float((rt * te).sum(1).mean()), 4)]
    import onnxruntime as ort
    result.update(variant=args.variant, provider=model._providers[0], onnxruntime=ort.__version__,
                  s_per_image=round(per_image, 4), s_per_caption=round(per_caption, 4))
    out = BASE / f"onnx_siglip2_{args.variant}_{args.device}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
