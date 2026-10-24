"""Real SigLIP 2 search embedder. Run with: uv run pytest -m integration (downloads models once)."""

import urllib.request
from pathlib import Path

import numpy as np
import pytest

from facegroup.adapters.media.decoder import PillowDecoder
from facegroup.adapters.models.siglip2_onnx import FP16, FP32, SigLip2
from facegroup.config import default_data_dir

SAMPLE_URL = "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def photo(pytestconfig) -> np.ndarray:
    path = Path(pytestconfig.cache.mkdir("samples")) / "t1.jpg"
    if not path.exists():
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return PillowDecoder().decode(path).rgb


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_group_photo_matches_its_description(photo, device):
    m = SigLip2(default_data_dir() / "models", device=device)
    img = m.embed_images([photo])[0]
    queries = ["a group of people posing for a photo", "a bowl of fruit", "a car on a highway", "a cat"]
    scores = m.embed_queries(queries) @ img
    assert int(np.argmax(scores)) == 0
    assert abs(np.linalg.norm(img) - 1) < 1e-4


def test_fp16_and_fp32_vectors_are_interchangeable(photo):
    a = SigLip2(default_data_dir() / "models", device="auto", variant=FP16)
    b = SigLip2(default_data_dir() / "models", device="cpu", variant=FP32)
    if a._providers[0] != "DmlExecutionProvider":
        pytest.skip("no GPU: fp16 is only used with DirectML")
    assert float(a.embed_images([photo])[0] @ b.embed_images([photo])[0]) > 0.995
    q = "people smiling"
    assert float(a.embed_queries([q])[0] @ b.embed_queries([q])[0]) > 0.995
