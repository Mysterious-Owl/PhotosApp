from __future__ import annotations

from typing import Protocol

import numpy as np

from facegroup.domain.entities import DetectedFace


class FaceAnalyzer(Protocol):
    """Detects faces in an RGB image and returns them with L2-normalised embeddings."""

    detector_id: str
    model_id: str  # identifies the embedding space; stored with every vector

    def analyze(self, image_rgb: np.ndarray) -> list[DetectedFace]: ...

    def close(self) -> None: ...
