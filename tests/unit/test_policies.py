import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from facegroup.domain.entities import DetectedFace, FileInfo, FileState, MediaRecord
from facegroup.domain.policies import Change, QualityGate, classify_change, iou, match_boxes, move_candidates


def rec(state=FileState.ACTIVE, size=100, mtime=1.0, content_hash="h"):
    return MediaRecord(1, 1, "a.jpg", size, mtime, state, content_hash, None)


def info(size=100, mtime=1.0, online_only=False):
    return FileInfo("a.jpg", size, mtime, online_only)


class TestClassifyChange:
    def test_unchanged(self):
        assert classify_change(rec(), info()) == Change.UNCHANGED

    def test_size_or_mtime_change(self):
        assert classify_change(rec(), info(size=101)) == Change.CHANGED
        assert classify_change(rec(), info(mtime=2.0)) == Change.CHANGED

    def test_online_only_wins(self):
        assert classify_change(rec(), info(online_only=True)) == Change.ONLINE_ONLY

    def test_hydrated(self):
        assert classify_change(rec(FileState.ONLINE_ONLY), info()) == Change.HYDRATED

    def test_missing_file_found_again(self):
        assert classify_change(rec(FileState.MISSING), info()) == Change.REAPPEARED
        assert classify_change(rec(FileState.MISSING), info(size=5)) == Change.CHANGED

    def test_changed_record_with_outstanding_reindex_is_not_requeued(self):
        # The CHANGED record already stores the new size/mtime; a rescan must not reset its pending job.
        assert classify_change(rec(FileState.CHANGED), info()) == Change.UNCHANGED


class TestMoveCandidates:
    def test_needs_same_size_and_stored_hash(self):
        missing = [rec(size=100), rec(size=100, content_hash=None), rec(size=7)]
        assert move_candidates(info(), missing) == [missing[0]]

    def test_never_for_online_only_files(self):
        assert move_candidates(info(online_only=True), [rec()]) == []


class TestBoxes:
    def test_iou(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
        assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9

    def test_match_boxes_prefers_best_overlap(self):
        old = [(0, 0, 10, 10), (100, 100, 110, 110)]
        new = [(101, 101, 111, 111), (1, 1, 11, 11), (500, 500, 510, 510)]
        assert match_boxes(old, new) == {0: 1, 1: 0}

    @given(st.lists(st.tuples(st.integers(0, 50), st.integers(0, 50), st.integers(1, 30), st.integers(1, 30)),
                    max_size=6),
           st.lists(st.tuples(st.integers(0, 50), st.integers(0, 50), st.integers(1, 30), st.integers(1, 30)),
                    max_size=6))
    def test_match_is_one_to_one_and_above_threshold(self, a, b):
        old = [(x, y, x + w, y + h) for x, y, w, h in a]
        new = [(x, y, x + w, y + h) for x, y, w, h in b]
        m = match_boxes(old, new)
        assert len(set(m.values())) == len(m)
        assert all(iou(old[o], new[n]) >= 0.5 for n, o in m.items())


class TestQualityGate:
    def face(self, side=100, score=0.9, blur=150.0):
        return DetectedFace((0, 0, side, side), np.zeros((5, 2), np.float32), score, blur=blur)

    def test_rejects_small_and_unsure(self):
        gate = QualityGate(min_face_px=40, min_det_score=0.5)
        assert gate.accepts(self.face())
        assert not gate.accepts(self.face(side=30))
        assert not gate.accepts(self.face(score=0.3))

    def test_score_rewards_size_and_sharpness(self):
        big_sharp = QualityGate.score(self.face(side=150, blur=300))
        small_blurry = QualityGate.score(self.face(side=50, blur=20))
        assert 0 < small_blurry < big_sharp <= 1
