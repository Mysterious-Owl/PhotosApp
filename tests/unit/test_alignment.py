import numpy as np

from facegroup.adapters.models.buffalo_onnx import ARCFACE_DST, nms, norm_crop, umeyama


def test_umeyama_recovers_a_known_similarity_transform():
    angle, scale, tx, ty = np.deg2rad(20), 1.7, 30.0, -12.0
    r = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = np.random.default_rng(0).uniform(0, 100, (5, 2))
    dst = src @ r.T + [tx, ty]
    t = umeyama(src, dst)
    assert np.allclose(t[:2, :2], r, atol=1e-9)
    assert np.allclose(t[:2, 2], [tx, ty], atol=1e-9)


def test_norm_crop_puts_landmarks_on_the_arcface_template():
    # Draw dots at landmark positions in a big image, then check they land on the template after alignment.
    img = np.zeros((600, 600, 3), np.uint8)
    lm = ARCFACE_DST * 3 + [150, 120]
    for x, y in lm.astype(int):
        img[y - 1 : y + 2, x - 1 : x + 2] = 255
    crop = norm_crop(img, lm.astype(np.float32))
    assert crop.shape == (112, 112, 3)
    for x, y in ARCFACE_DST.round().astype(int):
        assert crop[y - 1 : y + 2, x - 1 : x + 2].max() > 100


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.4) == [0, 2]
