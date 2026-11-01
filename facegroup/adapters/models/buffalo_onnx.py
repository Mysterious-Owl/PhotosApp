"""InsightFace buffalo_l (SCRFD-10GF detector + ArcFace R50 on WebFace600K) run directly with onnxruntime.

Re-implements the small parts of the `insightface` package we need (SCRFD decoding, NMS, 5-point
alignment), which avoids its C++ build requirement on Windows. Weights are licensed by InsightFace for
non-commercial research use; FaceGroup is personal use only (see PLAN.md §7).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from facegroup.adapters.models.download import download
from facegroup.domain.entities import DetectedFace

log = logging.getLogger(__name__)
ort.set_default_logger_severity(3)

BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"

# ArcFace canonical 5-point template for a 112x112 crop.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def ensure_buffalo_l(models_dir: Path, progress=None) -> Path:
    """Download buffalo_l once and keep only the two files we use. Returns the model folder."""
    target = models_dir / "buffalo_l"
    # MODEL_INFO.json is written last, so its presence means a complete install (not one cut off mid-extract).
    if all((target / f).exists() for f in (DET_FILE, REC_FILE, "MODEL_INFO.json")):
        return target
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
        zip_path = download(BUFFALO_L_URL, Path(tmp) / "buffalo_l.zip", progress)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base in (DET_FILE, REC_FILE):
                    part = target / f"{base}.part"
                    with zf.open(name) as src, open(part, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    part.replace(target / base)
    missing = [f for f in (DET_FILE, REC_FILE) if not (target / f).exists()]
    if missing:
        raise RuntimeError(f"buffalo_l archive did not contain {missing}")
    info = {f: _sha256(target / f) for f in (DET_FILE, REC_FILE)}
    info["source"] = BUFFALO_L_URL
    (target / "MODEL_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) mapping src to dst. 3x3."""
    n, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    a = dst_d.T @ src_d / n
    d = np.ones(dim)
    if np.linalg.det(a) < 0:
        d[-1] = -1
    u, s, vt = np.linalg.svd(a)
    t = np.eye(dim + 1)
    t[:dim, :dim] = u @ np.diag(d) @ vt
    scale = (s @ d) / src_d.var(axis=0).sum()
    t[:dim, dim] = dst_mean - scale * (t[:dim, :dim] @ src_mean)
    t[:dim, :dim] *= scale
    return t


def norm_crop(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    m = umeyama(landmarks.astype(np.float64), ARCFACE_DST.astype(np.float64) * (size / 112.0))[:2]
    return cv2.warpAffine(image, m, (size, size), borderValue=0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][overlap <= threshold]
    return keep


def choose_providers(device: str) -> list[str]:
    """'auto' prefers DirectML (any Windows GPU, including integrated) and falls back to the CPU."""
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "directml" and "DmlExecutionProvider" not in available:
        raise RuntimeError("DirectML requested but onnxruntime-directml is not installed")
    if device in ("auto", "directml") and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class BuffaloL:
    detector_id = "scrfd_10g@buffalo_l"
    model_id = "arcface_r50_w600k@buffalo_l"

    def __init__(self, model_dir: Path, threads: int = 0, device: str = "auto", det_size: int = 640,
                 det_threshold: float = 0.5, nms_threshold: float = 0.4) -> None:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # the recogniser's output shape metadata says batch 1; batches work on CPU
        if threads:  # 0 lets ONNX Runtime pick (best on hybrid P/E-core CPUs)
            opts.intra_op_num_threads = threads
        providers = choose_providers(device)
        if providers[0] == "DmlExecutionProvider":
            opts.enable_mem_pattern = False  # required by DirectML
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.provider = providers[0]
        self._det = ort.InferenceSession(str(model_dir / DET_FILE), opts, providers=providers)
        self._rec = ort.InferenceSession(str(model_dir / REC_FILE), opts, providers=providers)
        self._det_input = self._det.get_inputs()[0].name
        self._rec_input = self._rec.get_inputs()[0].name
        # DirectML enforces the declared output shape, so feed it one face at a time.
        self._rec_batch = self.provider == "CPUExecutionProvider"
        self._det_size = det_size
        self._det_threshold = det_threshold
        self._nms_threshold = nms_threshold
        self._strides = (8, 16, 32)
        self._num_anchors = 2
        self._centers: dict[tuple[int, int], np.ndarray] = {}

    def close(self) -> None:
        self._det = None
        self._rec = None

    # -- public -----------------------------------------------------------------------------------

    def analyze(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        faces = self._detect(image_rgb)
        if not faces:
            return self._analyze_padded(image_rgb)
        return self._describe(image_rgb, faces)

    def _analyze_padded(self, image_rgb: np.ndarray, pad_frac: float = 0.25) -> list[DetectedFace]:
        """Retry with a border. SCRFD misses faces that fill the frame and touch its edges (close-up
        selfies, tight crops): on LFW this recovered all 11 of 7,701 images with no detection."""
        h, w = image_rgb.shape[:2]
        p = int(max(h, w) * pad_frac)
        padded = cv2.copyMakeBorder(image_rgb, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = self._detect(padded)
        if not faces:
            return []
        faces = self._describe(padded, faces)  # embeddings from the padded frame: same black fill as warpAffine
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            f.bbox = (max(0.0, x1 - p), max(0.0, y1 - p), min(float(w), x2 - p), min(float(h), y2 - p))
            f.landmarks = f.landmarks - p
        return faces

    def _describe(self, image_rgb: np.ndarray, faces: list[DetectedFace]) -> list[DetectedFace]:
        crops = [norm_crop(image_rgb, f.landmarks) for f in faces]
        for face, crop in zip(faces, crops):
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            face.blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        for face, emb in zip(faces, self._embed(crops)):
            face.embedding = emb
        return faces

    # -- detection (SCRFD) ------------------------------------------------------------------------

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, stride)
        if key not in self._centers:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            self._centers[key] = np.repeat(centers, self._num_anchors, axis=0)
        return self._centers[key]

    def _detect(self, img: np.ndarray) -> list[DetectedFace]:
        size = self._det_size
        h, w = img.shape[:2]
        if h / w > 1:
            new_h, new_w = size, max(1, int(size * w / h))
        else:
            new_w, new_h = size, max(1, int(size * h / w))
        scale = new_h / h
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self._det.run(None, {self._det_input: blob})

        fmc = len(self._strides)
        all_scores, all_boxes, all_kps = [], [], []
        for idx, stride in enumerate(self._strides):
            scores = outs[idx].reshape(-1)
            boxes = outs[idx + fmc].reshape(-1, 4) * stride
            kps = outs[idx + 2 * fmc].reshape(-1, 10) * stride
            centers = self._anchor_centers(size // stride, size // stride, stride)
            pos = np.where(scores >= self._det_threshold)[0]
            if pos.size == 0:
                continue
            c = centers[pos]
            b = boxes[pos]
            all_boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=1))
            k = kps[pos].reshape(-1, 5, 2) + c[:, None, :]
            all_kps.append(k)
            all_scores.append(scores[pos])
        if not all_scores:
            return []
        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale
        keep = nms(boxes, scores, self._nms_threshold)
        out: list[DetectedFace] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            out.append(
                DetectedFace(
                    bbox=(float(max(0, x1)), float(max(0, y1)), float(min(w, x2)), float(min(h, y2))),
                    landmarks=kps[i].astype(np.float32),
                    det_score=float(scores[i]),
                )
            )
        return out

    # -- recognition (ArcFace) ----------------------------------------------------------------------

    def _embed(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        blob = ((np.stack(crops).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
        if self._rec_batch:
            feats = self._rec.run(None, {self._rec_input: blob})[0]
        else:
            feats = np.concatenate([self._rec.run(None, {self._rec_input: b[None]})[0] for b in blob])
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return list((feats / np.maximum(norms, 1e-12)).astype(np.float32))
