"""Difference hash (dHash): a 64-bit fingerprint of an image's light/dark layout. Survives resizing and
re-compression, changes with crops, so it finds copies of the same picture."""

from __future__ import annotations

import cv2
import numpy as np


def dhash(rgb: np.ndarray) -> int:
    """Computed on a 320-px version so that the working image and the cached thumbnail give the same hash."""
    h, w = rgb.shape[:2]
    scale = 320 / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def to_signed(value: int) -> int:
    """SQLite integers are signed 64-bit."""
    return value - (1 << 64) if value >= 1 << 63 else value


def from_signed(value: int) -> int:
    return value + (1 << 64) if value < 0 else value
