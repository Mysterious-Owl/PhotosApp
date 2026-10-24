from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facegroup.bootstrap import App, build
from facegroup.config import Settings
from facegroup.domain.entities import DetectedFace


class FakeAnalyzer:
    """Deterministic stand-in for a face model.

    Finds one 'face' in the top-left corner of any image at least 120 px on both sides. Its embedding is
    derived from the corner's mean colour, so the same photo always yields the same vector.
    """

    detector_id = "fake-detector"
    model_id = "fake-embedder"
    loads = 0

    def __init__(self, fail_on_color: tuple[int, int, int] | None = None) -> None:
        FakeAnalyzer.loads += 1
        self.fail_on_color = fail_on_color
        self.closed = False

    def analyze(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        h, w = image_rgb.shape[:2]
        if h < 120 or w < 120:
            return []
        corner = image_rgb[10:110, 10:110].reshape(-1, 3).mean(axis=0)
        if self.fail_on_color and np.allclose(corner, self.fail_on_color, atol=3):
            raise RuntimeError("simulated model failure")
        rng = np.random.default_rng(int(corner.sum() * 1000))
        emb = rng.standard_normal(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        landmarks = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float32)
        return [DetectedFace(bbox=(10, 10, 110, 110), landmarks=landmarks, det_score=0.9, embedding=emb, blur=150)]

    def close(self) -> None:
        self.closed = True


class FakeEmbedder:
    """Deterministic stand-in for an image-text model. An image's vector encodes its mean colour; a text query
    names a colour ("red", "green", "blue"), so search results can be checked exactly."""

    model_id = "fake-embedder-v1"
    dim = 8
    loads = 0
    COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "grey": (128, 128, 128)}

    def __init__(self) -> None:
        FakeEmbedder.loads += 1

    @staticmethod
    def _vec(rgb) -> np.ndarray:
        r, g, b = (float(x) / 255 for x in rgb)
        v = np.array([r, g, b, 1 - r, 1 - g, 1 - b, 0.1, 0.1], np.float32)
        return v / np.linalg.norm(v)

    def embed_images(self, images):
        return np.stack([self._vec(im.reshape(-1, 3).mean(axis=0)) for im in images])

    def embed_queries(self, texts):
        return np.stack([self._vec(self.COLOURS.get(t.strip().lower(), (128, 128, 128))) for t in texts])

    def close(self) -> None:
        pass


def make_photo(path: Path, size=(400, 300), color=(200, 120, 80), exif_orientation: int | None = None,
               fmt: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    img.save(path, format=fmt, **kwargs)
    return path


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    # Never read the user's real config, and never treat the test folder as synced.
    monkeypatch.setenv("FACEGROUP_DATA_DIR", str(tmp_path / "data"))
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(name, raising=False)
    FakeAnalyzer.loads = 0


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", backoff_seconds=0, decode_workers=2, batch_size=4,
                    min_face_px=40)


@pytest.fixture
def library_dir(tmp_path) -> Path:
    d = tmp_path / "library"
    d.mkdir()
    return d


@pytest.fixture
def make_app(settings):
    apps: list[App] = []

    def factory(analyzer=None, **overrides) -> App:
        s = settings.model_copy(update=overrides) if overrides else settings
        a = build(s, analyzer_factory=analyzer or FakeAnalyzer, face_model_id=FakeAnalyzer.model_id,
                  embedder=FakeEmbedder, embed_model_id=FakeEmbedder.model_id)
        apps.append(a)
        return a

    yield factory
    for a in apps:
        try:
            a.close()
        except Exception:
            pass


@pytest.fixture
def unicode_name() -> str:
    return "फ़ोटो é 写真"


def count(app: App, sql: str, *args) -> int:
    from facegroup.adapters.storage.db import connect

    conn = connect(app.settings.db_path)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


def write_sql(app: App, sql: str, *args) -> None:
    app._writer.submit(lambda c: c.execute(sql, args))
