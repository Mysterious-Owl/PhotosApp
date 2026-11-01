"""Image-text retrieval on Flickr30k 1K test, using each model's official reference implementation.

Runs in a separate reference environment (PyTorch + transformers), not the app's:
    data\\eval\\refenv\\Scripts\\python.exe tests/eval/eval_retrieval.py [models...]

1,000 images x 5 human captions. Text->image: for each caption, is its image in the top K? Image->text: for
each image, is one of its captions in the top K? Same data, same metric, same machine for every model.
Data (~141 MB, Flickr terms: research use) is downloaded to data\\eval\\flickr1k.
"""

from __future__ import annotations

import ast
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

BASE = Path(__file__).resolve().parents[2] / "data" / "eval"  # runs in the reference env, without facegroup
DATA = BASE / "flickr1k"
REPO = "https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main"


def fetch() -> tuple[list[Path], list[str], np.ndarray]:
    DATA.mkdir(parents=True, exist_ok=True)
    csv = DATA / "test_1k_flickr.csv"
    if not csv.exists():
        urllib.request.urlretrieve(f"{REPO}/test_1k_flickr.csv", csv)
    img_dir = DATA / "images"
    if not img_dir.exists():
        z = DATA / "images.zip"
        if not z.exists():
            urllib.request.urlretrieve(f"{REPO}/images_flickr_1k_test.zip", z)
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():  # untrusted archive: keep plain file names only
                name = Path(member).name
                if name.lower().endswith(".jpg"):
                    img_dir.mkdir(exist_ok=True)
                    (img_dir / name).write_bytes(zf.read(member))
    df = pd.read_csv(csv)
    images, captions, owner = [], [], []
    for i, row in df.iterrows():
        images.append(img_dir / Path(row["filename"]).name)
        caps = ast.literal_eval(row["raw"]) if isinstance(row["raw"], str) else row["raw"]
        for c in caps[:5]:
            captions.append(c)
            owner.append(i)
    return images, captions, np.array(owner)


def recall(sim: np.ndarray, owner: np.ndarray) -> dict[str, float]:
    """sim: (n_captions, n_images)."""
    out = {}
    rank_t2i = (-sim).argsort(axis=1)
    for k in (1, 5, 10):
        out[f"t2i_R@{k}"] = float((rank_t2i[:, :k] == owner[:, None]).any(axis=1).mean() * 100)
    rank_i2t = (-sim.T).argsort(axis=1)
    for k in (1, 5, 10):
        hit = [(owner[rank_i2t[i, :k]] == i).any() for i in range(sim.shape[1])]
        out[f"i2t_R@{k}"] = float(np.mean(hit) * 100)
    return out


# -- models (official implementations) ---------------------------------------------------------------

def embeddinggemma2():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": torch.float32},
                            config_kwargs={"audio_config": None})

    def img(batch):
        return m.encode([{"image": im} for im in batch], normalize_embeddings=True, batch_size=8)

    def txt(texts):
        return m.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True, batch_size=64)
    return img, txt


def siglip2(name="google/siglip2-base-patch16-224"):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(name)

    @torch.no_grad()
    def img(batch):
        f = model.get_image_features(**proc(images=batch, return_tensors="pt"))
        f = getattr(f, "pooler_output", f)
        return torch.nn.functional.normalize(f, dim=-1).numpy()

    @torch.no_grad()
    def txt(texts):
        out = []
        for i in range(0, len(texts), 256):
            enc = proc(text=[t.lower() for t in texts[i:i + 256]], padding="max_length", max_length=64,
                       truncation=True, return_tensors="pt")
            f = model.get_text_features(**enc)
            f = getattr(f, "pooler_output", f)
            out.append(torch.nn.functional.normalize(f, dim=-1).numpy())
        return np.concatenate(out)
    return img, txt


def openclip_b32():
    from transformers import CLIPModel, CLIPProcessor
    name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def img(batch):
        f = model.get_image_features(**proc(images=batch, return_tensors="pt"))
        f = getattr(f, "pooler_output", f)
        return torch.nn.functional.normalize(f, dim=-1).numpy()

    @torch.no_grad()
    def txt(texts):
        out = []
        for i in range(0, len(texts), 256):
            f = model.get_text_features(**proc(text=texts[i:i + 256], padding=True, truncation=True, return_tensors="pt"))
            f = getattr(f, "pooler_output", f)
            out.append(torch.nn.functional.normalize(f, dim=-1).numpy())
        return np.concatenate(out)
    return img, txt


MODELS = {"embeddinggemma-2": embeddinggemma2, "siglip2-base": siglip2, "openclip-b32": openclip_b32}


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    images, captions, owner = fetch()
    print(f"{len(images)} images, {len(captions)} captions")
    results = {}
    out = BASE / "retrieval_results.json"
    if out.exists():
        results = json.loads(out.read_text())
    for name in names:
        t0 = time.monotonic()
        img_fn, txt_fn = MODELS[name]()
        load_s = time.monotonic() - t0
        t1 = time.monotonic()
        ie = []
        for i in range(0, len(images), 16):
            ie.append(img_fn([Image.open(p).convert("RGB") for p in images[i:i + 16]]))
            if i % 160 == 0:
                print(f"  {name}: {i + 16}/{len(images)} images", flush=True)
        ie = np.concatenate(ie)
        img_s = (time.monotonic() - t1) / len(images)
        t2 = time.monotonic()
        te = txt_fn(captions)
        txt_s = (time.monotonic() - t2) / len(captions)
        r = recall(te @ ie.T, owner)
        r.update(load_s=round(load_s, 1), torch_cpu_s_per_image=round(img_s, 3),
                 torch_cpu_s_per_caption=round(txt_s, 4), dim=int(ie.shape[1]))
        results[name] = r
        np.save(BASE / f"ref_{name}_images.npy", ie)
        np.save(BASE / f"ref_{name}_texts.npy", te)
        out.write_text(json.dumps(results, indent=2))
        print(name, json.dumps(r), flush=True)
        del img_fn, txt_fn


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
