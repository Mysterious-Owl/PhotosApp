"""Real buffalo_l on a real group photo. Run with: uv run pytest -m integration

Downloads the model (~280 MB, once) into the user's FaceGroup data folder and a sample photo from the
InsightFace repository into the pytest cache. Nothing is committed.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pytest

from facegroup.adapters.media.decoder import PillowDecoder
from facegroup.adapters.models.buffalo_onnx import BuffaloL, ensure_buffalo_l
from facegroup.config import default_data_dir

SAMPLE_URL = "https://raw.githubusercontent.com/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", params=["cpu", "auto"])
def model(request):
    m = BuffaloL(ensure_buffalo_l(default_data_dir() / "models"), device=request.param)
    yield m
    m.close()


@pytest.fixture(scope="module")
def group_photo(pytestconfig) -> Path:
    path = Path(pytestconfig.cache.mkdir("samples")) / "t1.jpg"
    if not path.exists():
        urllib.request.urlretrieve(SAMPLE_URL, path)
    return path


def test_detects_every_face_in_a_group_photo(model, group_photo):
    img = PillowDecoder().decode(group_photo).rgb
    faces = model.analyze(img)
    # t1.jpg is a group photo of 6 people.
    assert len(faces) == 6
    for f in faces:
        assert f.det_score > 0.5
        assert f.embedding.shape == (512,)
        assert abs(np.linalg.norm(f.embedding) - 1) < 1e-4
        x1, y1, x2, y2 = f.bbox
        lm = f.landmarks
        assert (lm[:, 0] >= x1 - 5).all() and (lm[:, 0] <= x2 + 5).all()  # landmarks sit inside the box
        assert lm[0, 0] < lm[1, 0]  # left eye left of right eye


def test_same_person_matches_across_scales_and_differs_from_others(model, group_photo):
    img = PillowDecoder().decode(group_photo).rgb
    faces = model.analyze(img)
    # Same photo at a smaller size: each face should match its counterpart far better than anyone else.
    small = PillowDecoder(max_side=max(img.shape[:2]) * 2 // 3).decode(group_photo).rgb
    faces_small = model.analyze(small)
    assert len(faces_small) == len(faces)
    a = np.stack([f.embedding for f in sorted(faces, key=lambda f: f.bbox[0])])
    b = np.stack([f.embedding for f in sorted(faces_small, key=lambda f: f.bbox[0])])
    sim = a @ b.T
    assert (np.diag(sim) > 0.7).all()
    off_diag = sim[~np.eye(len(faces), dtype=bool)]
    assert off_diag.max() < 0.4
