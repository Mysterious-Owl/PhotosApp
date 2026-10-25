"""OCR accuracy and speed on synthetic images with known text (never the user's library).

    uv run python tests/eval/eval_ocr.py

Word recall = share of the words drawn into the image that appear in the OCR output (case-insensitive).
"""

from __future__ import annotations

import random
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from tests.eval.eval_categories import WORDS, font  # noqa: E402

rng = random.Random(7)


def text_lines(n, k=7):
    return [" ".join(rng.choice(WORDS) for _ in range(k)).capitalize() for _ in range(n)]


def screenshot():
    lines = text_lines(12)
    im = Image.new("RGB", (1170, 2532), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 1170, 180], fill=(37, 211, 102))
    for i, line in enumerate(lines):
        y = 260 + i * 170
        d.rounded_rectangle([40, y, 1130, y + 120], 24, fill=(235, 235, 240))
        d.text((70, y + 35), line, fill=(20, 20, 30), font=font("segoeui.ttf", 40))
    return im, lines


def document():
    lines = text_lines(20, 9)
    im = Image.new("RGB", (1240, 1754), (252, 252, 250))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((100, 120 + i * 70), line, fill=(10, 10, 10), font=font("times.ttf", 30))
    return im.rotate(1.5, fillcolor=(120, 110, 100), expand=True), lines


def receipt():
    lines = [f"{rng.choice(WORDS).upper()} {rng.uniform(1, 60):.2f}" for _ in range(15)]
    im = Image.new("RGB", (600, 1100), (250, 250, 245))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((40, 60 + i * 60), line, fill=(0, 0, 0), font=font("consola.ttf", 30))
    return im, lines


def words(s):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())


def main():
    from facegroup.adapters.models.ocr import RapidOcrReader

    cases = [make() for make in (screenshot, document, receipt) for _ in range(8)]
    for device in ("cpu",):
        reader = RapidOcrReader(device=device)
        reader.read(np.asarray(cases[0][0]))  # warm-up
        hits = total = 0
        t0 = time.monotonic()
        per_kind: dict[str, list[float]] = {}
        for idx, (im, lines) in enumerate(cases):
            # Same as the app: OCR runs on the working image (longest side 2048).
            im = im.copy()
            im.thumbnail((2048, 2048))
            text = reader.read(np.asarray(im.convert("RGB")))
            got = set(words(text))
            want = [w for line in lines for w in words(line)]
            h = sum(w in got for w in want)
            hits, total = hits + h, total + len(want)
            per_kind.setdefault(("screenshot", "document", "receipt")[idx // 8], []).append(h / len(want))
        dt = (time.monotonic() - t0) / len(cases)
        print(f"{reader.provider}: word recall {100 * hits / total:.1f}% "
              + " ".join(f"{k} {100 * np.mean(v):.0f}%" for k, v in per_kind.items()) + f", {dt:.2f} s/image")


if __name__ == "__main__":
    main()
