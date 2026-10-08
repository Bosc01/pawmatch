import numpy as np
import pytest

retrieve = pytest.importorskip("pawmatch.retrieve")


def test_mnn_scores_hand_computed():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[[1.0, 0.0], [0.0, 1.0]],    # both query patches match mutually: 1 + 1
                  [[1.0, 0.0], [1.0, 0.0]]])   # only query patch 0 has a mutual match: 1
    assert retrieve.mnn_scores(a, b).tolist() == [2.0, 1.0]


def test_rrf_hand_computed():
    # Stage-2 scores reverse the stage-1 order of three candidates.
    fused = retrieve._rrf(np.array([[0.1, 0.2, 0.3]]))
    k = retrieve.RRF_K
    assert fused[0] == pytest.approx([1 / (k + 1) + 1 / (k + 3), 2 / (k + 2), 1 / (k + 3) + 1 / (k + 1)])


def test_true_rank():
    ids = np.array([[2, 0, 1], [1, 2, -1]])
    g_dogs = np.array(["a", "b", "c"])
    assert retrieve._true_rank(ids, np.array(["a", "z"]), g_dogs).tolist() == [2, 0]
