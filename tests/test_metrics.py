import numpy as np
import pytest

from pawmatch import metrics


def test_match_ranks_hand_computed():
    sim = np.array([[0.9, 0.1, 0.5],   # dog a: true 0.9 is the top score -> rank 1
                    [0.7, 0.2, 0.6],   # dog c: true 0.6, one higher -> rank 2
                    [0.4, 0.4, 0.1],   # dog b: true 0.4 ties 0.4, ties count against -> rank 2
                    [0.3, 0.2, 0.1]])  # dog z: no gallery photo -> 0
    ranks = metrics.match_ranks(sim, np.array(["a", "c", "b", "z"]), np.array(["a", "b", "c"]))
    assert ranks.tolist() == [1, 2, 2, 0]


def test_average_precision_hand_computed():
    sim = np.array([[0.9, 0.8, 0.7, 0.6], [0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]])
    rel = np.array([[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], bool)
    ap = metrics.average_precision(sim, rel)
    assert ap[0] == pytest.approx((1 / 1 + 2 / 3) / 2)  # hits at ranks 1 and 3
    assert ap[1] == pytest.approx(1 / 4)  # the only hit is ranked last
    assert np.isnan(ap[2])  # no relevant item


def test_verification_auc_and_paper_clamp():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.3, -0.3, -0.1, -0.5, -0.2])
    # Raw: 0.3 beats all three negatives, -0.3 beats only -0.5: 4 of 6 pairs.
    assert metrics.verification_auc(y, s) == pytest.approx(4 / 6)
    # Clamped as in PetFace compute_auc.py: positives 0.3 and 0, negatives all 0.
    # 3 wins plus 3 ties at one half each: 4.5 of 6.
    assert metrics.verification_auc(y, s, clamp_negative=True) == pytest.approx(4.5 / 6)


def test_bootstrap_ci():
    mean_of = lambda v: (lambda w: np.average(v, weights=w))
    constant = np.array([1.0, 1.0, 1.0])
    assert metrics.bootstrap_ci(mean_of(constant), np.array(["a", "a", "b"])) == [1.0, 1.0]
    # Two dogs scoring 0 and 1: resampled means are 0, 0.5, 1 with probability 1/4, 1/2, 1/4.
    v, g = np.array([0.0, 1.0]), np.array(["a", "b"])
    ci = metrics.bootstrap_ci(mean_of(v), g, n=2000, seed=1)
    assert ci == [0.0, 1.0]
    assert ci == metrics.bootstrap_ci(mean_of(v), g, n=2000, seed=1)
