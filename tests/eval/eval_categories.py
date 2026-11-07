"""Tune the clutter margin on public photos (Flickr30k 1K) and synthetic clutter. Never uses the user's library.

    uv run python tests/eval/eval_categories.py

Reports, for each margin: how many real photos would be wrongly called clutter, and how much synthetic clutter is
caught per category. The synthetic images are crude, so their recall is a sanity check, not a promise.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from facegroup.adapters.models.siglip2_onnx import SigLip2
from facegroup.config import default_data_dir, default_eval_dir
from facegroup.domain.organize import CATEGORY_PROMPTS, PHOTO, classify, prototypes

BASE = default_eval_dir()
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
rng = random.Random(0)


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


WORDS = ("meeting project update please review the attached file today tomorrow thanks hello team order total "
         "price payment delivery account settings profile message reply share download upload search home").split()


def sentence(n=8):
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize()


def screenshot() -> Image.Image:
    w, h = rng.choice([(1080, 2340), (1170, 2532), (1920, 1080), (1440, 900)])
    im = Image.new("RGB", (w, h), rng.choice([(255, 255, 255), (245, 246, 250), (30, 30, 36)]))
    d = ImageDraw.Draw(im)
    dark = im.getpixel((0, 0))[0] < 100
    fg = (230, 230, 235) if dark else (20, 20, 30)
    d.rectangle([0, 0, w, int(h * 0.07)], fill=rng.choice([(0, 120, 215), (37, 211, 102), (88, 101, 242), (60, 60, 70)]))
    f = font("segoeui.ttf", max(18, w // 40))
    y = int(h * 0.09)
    while y < h * 0.95:
        if rng.random() < 0.5:  # chat bubble
            x0 = rng.choice([int(w * 0.05), int(w * 0.35)])
            d.rounded_rectangle([x0, y, x0 + int(w * 0.6), y + int(h * 0.05)], 20,
                                fill=(220, 248, 198) if x0 > w * 0.2 else (235, 235, 240))
            d.text((x0 + 20, y + 10), sentence(6), fill=(20, 20, 30), font=f)
        else:
            d.text((int(w * 0.05), y), sentence(9), fill=fg, font=f)
        y += int(h * 0.07)
    return im


def document() -> Image.Image:
    im = Image.new("RGB", (1240, 1754), (252, 252, 250))
    d = ImageDraw.Draw(im)
    d.text((100, 100), sentence(5).upper(), fill=(0, 0, 0), font=font("timesbd.ttf", 44))
    f = font("times.ttf", 26)
    y = 200
    while y < 1650:
        d.text((100, y), sentence(14), fill=(10, 10, 10), font=f)
        y += 38 if rng.random() > 0.1 else 70
    return im.rotate(rng.uniform(-3, 3), fillcolor=(120, 110, 100), expand=True)


def receipt() -> Image.Image:
    im = Image.new("RGB", (600, 1500), (250, 250, 245))
    d = ImageDraw.Draw(im)
    f = font("consola.ttf", 24)
    d.text((150, 40), "SUPER MART", fill=(0, 0, 0), font=font("consolab.ttf", 40))
    y, total = 140, 0.0
    for _ in range(rng.randint(10, 22)):
        price = rng.uniform(1, 60)
        total += price
        d.text((40, y), f"{rng.choice(WORDS).upper():<18}{price:>8.2f}", fill=(0, 0, 0), font=f)
        y += 40
    d.text((40, y + 30), f"{'TOTAL':<18}{total:>8.2f}", fill=(0, 0, 0), font=font("consolab.ttf", 26))
    bg = Image.new("RGB", (900, 1700), rng.choice([(90, 70, 50), (40, 40, 45), (160, 150, 140)]))
    bg.paste(im.rotate(rng.uniform(-6, 6), expand=True, fillcolor=bg.getpixel((0, 0))), (120, 60))
    return bg


def meme(photo: Path) -> Image.Image:
    im = Image.open(photo).convert("RGB").resize((800, 600))
    d = ImageDraw.Draw(im)
    f = font("impact.ttf", 64)
    for text, y in ((sentence(4).upper(), 20), (sentence(4).upper(), 500)):
        d.text((400, y), text, fill=(255, 255, 255), font=f, anchor="ma", stroke_width=4, stroke_fill=(0, 0, 0))
    return im


def illustration() -> Image.Image:
    w, h = 1920, 1080
    top, bottom = [rng.randint(0, 255) for _ in range(3)], [rng.randint(0, 255) for _ in range(3)]
    grad = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(top) * (1 - grad) + np.array(bottom) * grad).astype(np.uint8)
    im = Image.fromarray(np.repeat(arr, w, axis=1))
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(5, 15)):
        x, y, r = rng.randint(0, w), rng.randint(0, h), rng.randint(40, 300)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return im


def main() -> None:
    model = SigLip2(default_data_dir() / "models")
    protos = prototypes({c: model.embed_queries(p) for c, p in CATEGORY_PROMPTS.items()})

    with open(BASE / "flickr1k" / "test_1k_flickr.csv", encoding="utf-8", newline="") as f:
        photos = [BASE / "flickr1k" / "images" / Path(r["filename"]).name for r in csv.DictReader(f)]
    photo_vecs = np.concatenate([model.embed_images([np.asarray(Image.open(p).convert("RGB")) for p in photos[i:i + 16]])
                                 for i in range(0, len(photos), 16)])

    makers = {"screenshot": screenshot, "document": document, "receipt": receipt,
              "meme": lambda: meme(rng.choice(photos)), "illustration": illustration}
    synth = {c: model.embed_images([np.asarray(make().convert("RGB")) for _ in range(40)]) for c, make in makers.items()}

    print(f"{'margin':>7} {'photos wrongly clutter':>24} " + " ".join(f"{c:>12}" for c in makers))
    for margin in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        fp = sum(r.category != PHOTO for r in classify(photo_vecs, protos, margin)) / len(photos)
        recalls = []
        for c, v in synth.items():
            res = classify(v, protos, margin)
            recalls.append(sum(r.category == c for r in res) / len(res))
        print(f"{margin:>7.2f} {100 * fp:>23.1f}% " + " ".join(f"{100 * r:>11.0f}%" for r in recalls))


if __name__ == "__main__":
    main()
