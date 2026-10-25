"""Text in images (OCR) with RapidOCR: PaddleOCR PP-OCRv6 small detection + recognition, Apache-2.0, models bundled
in the package (no download). Runs on onnxruntime; uses the same CPU/DirectML choice as the other models."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class RapidOcrReader:
    model_id = "ppocrv6-small@rapidocr"

    def __init__(self, device: str = "cpu", min_score: float = 0.5, det_side: int = 960) -> None:
        from rapidocr import RapidOCR

        from facegroup.adapters.models.buffalo_onnx import choose_providers

        self.provider = choose_providers(device)[0]
        use_dml = self.provider == "DmlExecutionProvider"
        # Measured (docs/EVALUATION.md §8): text detection dominates; capping the detector's input at 960 px and
        # skipping the rotation classifier cut time ~35% with no loss on test images. CPU beat DirectML here.
        params = {"Global.log_level": "error", "Global.text_score": min_score, "Global.use_cls": False,
                  "Det.limit_side_len": det_side, "Det.limit_type": "max",
                  "EngineConfig.onnxruntime.use_dml": use_dml}
        try:
            self._engine = RapidOCR(params=params)
        except Exception:  # an engine option this RapidOCR build does not know: fall back to its defaults
            log.warning("RapidOCR rejected options %s; using defaults", params, exc_info=True)
            self._engine = RapidOCR()

    def read(self, image_rgb: np.ndarray) -> str:
        """All text found, one line per detected text box, top to bottom."""
        result = self._engine(image_rgb[:, :, ::-1].copy())  # RapidOCR expects BGR like OpenCV
        texts = getattr(result, "txts", None)
        if not texts:
            return ""
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) == len(texts):
            order = np.argsort([float(np.min(np.asarray(b)[:, 1])) for b in boxes], kind="stable")
            texts = [texts[i] for i in order]
        return "\n".join(t.strip() for t in texts if t and t.strip())

    def close(self) -> None:
        self._engine = None
