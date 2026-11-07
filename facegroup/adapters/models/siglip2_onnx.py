"""SigLIP 2 (base, patch 16, 224 px) image-text embedder, run with onnxruntime. Apache-2.0 weights.

Preprocessing matches the reference SiglipProcessor exactly (checked against transformers in
tests/eval/eval_retrieval_onnx.py): images resized to 224x224 with PIL bilinear, scaled to [-1, 1]; text
lower-cased, Gemma-tokenised with an end token, padded to 64.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from facegroup.adapters.models.download import download
from facegroup.adapters.models.onnx import choose_providers, session_options

log = logging.getLogger(__name__)
ort.set_default_logger_severity(3)

REPO = "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main"
# Checked against the reference on Flickr30k 1K (docs/EVALUATION.md §6): fp32 and fp16 match it (cosine >= 0.9988,
# recall within 0.2 points); the published int8 files do NOT (recall 78% -> 28-36%) and must not be used.
FP32, FP16 = "", "fp16"


def variant_for(providers: list[str]) -> str:
    """fp16 on a GPU (DirectML: faster, half the memory); fp32 on CPU, where fp16 is slow or unsupported."""
    return FP16 if providers[0] == "DmlExecutionProvider" else FP32


def ensure_siglip2(models_dir: Path, variant: str = FP32) -> Path:
    target = models_dir / "siglip2-base"
    target.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", f"onnx/{_file('vision', variant)}", f"onnx/{_file('text', variant)}"):
        dest = target / Path(f).name
        if not dest.exists():
            download(f"{REPO}/{f}", dest)
    return target


class SigLip2:
    # fp32 and fp16 vectors are interchangeable (cosine 0.999), so precision is not part of the id:
    # switching between GPU and CPU never forces a re-index.
    model_id = "siglip2-base-p16-224"
    dim = 768
    size = 224
    max_tokens = 64

    def __init__(self, models_dir: Path, threads: int = 0, device: str = "auto", variant: str | None = None) -> None:
        self._threads = threads
        self._providers = choose_providers(device)
        self._variant = variant_for(self._providers) if variant is None else variant
        self._dir = ensure_siglip2(models_dir, self._variant)
        self._vision: ort.InferenceSession | None = None
        self._text: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._load_lock = threading.Lock()  # warm-up, searches and organize can all ask first

    def _session(self, name: str) -> ort.InferenceSession:
        return ort.InferenceSession(str(self._dir / _file(name, self._variant)),
                                    session_options(self._providers, self._threads), providers=self._providers)

    # Each side loads on first use, so the web process only ever holds the text side.

    def embed_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        if self._vision is None:
            with self._load_lock:
                if self._vision is None:
                    self._vision = self._session("vision")
        batch = np.stack([self._preprocess(im) for im in images_rgb])
        out = self._vision.run(["pooler_output"], {"pixel_values": batch})[0]
        return _normalise(out)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if self._text is None:
            with self._load_lock:
                if self._text is None:
                    tokenizer = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
                    tokenizer.enable_truncation(self.max_tokens)
                    tokenizer.enable_padding(length=self.max_tokens, pad_id=0, pad_token="<pad>")
                    self._tokenizer = tokenizer
                    self._text = self._session("text")  # set last: other threads test it without the lock
        ids = np.array([e.ids for e in self._tokenizer.encode_batch([t.lower() for t in texts])], np.int64)
        out = self._text.run(["pooler_output"], {"input_ids": ids})[0]
        return _normalise(out)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        im = Image.fromarray(rgb).resize((self.size, self.size), Image.Resampling.BILINEAR)
        x = np.asarray(im, np.float32) / 255.0
        return ((x - 0.5) / 0.5).transpose(2, 0, 1)

    def close(self) -> None:
        self._vision = None
        self._text = None


def _file(side: str, variant: str) -> str:
    """onnx-community naming: vision_model.onnx (fp32), vision_model_quantized.onnx, vision_model_fp16.onnx."""
    return f"{side}_model{'_' + variant if variant else ''}.onnx"


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
