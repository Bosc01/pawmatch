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


def test_paired_permutation_p():
    # Three dogs with per-dog differences 1, 1, 0: |sum of signed differences| >= 2
    # in exactly half of the 2^3 sign patterns, so the exact p-value is 0.5.
    p = metrics.paired_permutation_p([1, 1, 0], [0, 0, 0], ["x", "y", "z"], n=20000)
    assert p == pytest.approx(0.5, abs=0.02)
    assert metrics.paired_permutation_p([1, 0], [1, 0], ["x", "y"]) == 1.0


def test_benjamini_hochberg_hand_computed():
    # Sorted p: 0.01, 0.03, 0.04, 0.20 -> times 4/rank: 0.04, 0.06, 0.0533, 0.20 -> running min from the top.
    q = metrics.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_threshold_at_far():
    impostors = np.arange(1000) / 1000  # 0.000 .. 0.999
    t, resolution = metrics.threshold_at_far(impostors, 0.01)
    assert (impostors >= t).sum() == 10 and resolution == 1 / 1000
    t, _ = metrics.threshold_at_far(impostors, 0.0001)  # below resolution: accept none
    assert (impostors >= t).sum() == 0


def test_threshold_for_gallery():
    impostors = np.arange(1000) / 1000
    t, per_impostor, _ = metrics.threshold_for_gallery(impostors, 0.01, 1)
    assert per_impostor == pytest.approx(0.01) and (impostors >= t).sum() == 10
    # Two gallery entries: 1 - sqrt(0.99) = 0.0050126 per impostor, so 5 of 1000 may pass.
    t, per_impostor, _ = metrics.threshold_for_gallery(impostors, 0.01, 2)
    assert per_impostor == pytest.approx(1 - 0.99 ** 0.5) and (impostors >= t).sum() == 5
