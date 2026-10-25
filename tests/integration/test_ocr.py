"""Real OCR (RapidOCR, PP-OCRv6 small, models bundled) on a synthetic image. Run with: uv run pytest -m integration"""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from facegroup.adapters.models.ocr import RapidOcrReader

pytestmark = pytest.mark.integration


def test_reads_text_from_a_synthetic_screenshot():
    im = Image.new("RGB", (1600, 900), (255, 255, 255))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 48)
    except OSError:
        f = ImageFont.load_default()
    d.text((60, 100), "Invoice number 48213", fill=(0, 0, 0), font=f)
    d.text((60, 300), "Delivery tomorrow morning", fill=(0, 0, 0), font=f)
    text = RapidOcrReader().read(np.asarray(im)).lower()
    for word in ("invoice", "48213", "delivery", "tomorrow"):
        assert word in text
    assert RapidOcrReader().read(np.full((400, 400, 3), 128, np.uint8)) == ""  # nothing to read
