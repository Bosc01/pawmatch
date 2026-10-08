"""Retrieval and verification metrics. Tests pin these to hand-computed values."""
import math

import numpy as np
from sklearn.metrics import roc_auc_score


def match_ranks(sim, q_dogs, g_dogs):
    """1-based rank of each query's true gallery photo, or 0 if its dog has none.

    sim is (queries, gallery). Ties count against the true match. Assumes at most
    one gallery photo per dog.
    """
    col = {d: j for j, d in enumerate(g_dogs)}
    has = np.array([d in col for d in q_dogs])
    true = sim[np.arange(len(sim)), [col.get(d, 0) for d in q_dogs]]
    return np.where(has, (sim >= true[:, None]).sum(1), 0)


def average_precision(sim, rel):
    """AP per row of a (queries, candidates) score matrix; nan for rows with no relevant item."""
    order = np.argsort(-sim, axis=1, kind="stable")
    r = np.take_along_axis(rel, order, 1)
    precision = np.cumsum(r, 1) / np.arange(1, r.shape[1] + 1)
    with np.errstate(invalid="ignore"):
        return (precision * r).sum(1) / r.sum(1)


def verification_auc(labels, sims, clamp_negative=False, sample_weight=None):
    """ROC AUC of same-dog pairs. clamp_negative reproduces PetFace's compute_auc.py,
    which sets negative cosine similarities to 0 before scoring."""
    return roc_auc_score(labels, np.maximum(sims, 0) if clamp_negative else sims, sample_weight=sample_weight)


def bootstrap_ci(stat, groups, n=1000, seed=0):
    """95% percentile interval of stat(weights), resampling groups (dogs) with replacement.

    A resample is expressed as per-item integer weights: how many times the item's
    group was drawn. That is exact for means and for sample-weighted AUC.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(seed).multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n)
    return np.percentile([stat(w[inv]) for w in draws], [2.5, 97.5]).tolist()


def paired_permutation_p(a, b, groups, n=10000, seed=0):
    """Two-sided p-value for mean(a - b) = 0: a sign-flip permutation test that flips
    each group's summed difference, so dogs, not photos, are the exchangeable units."""
    _, inv = np.unique(groups, return_inverse=True)
    d = np.bincount(inv, weights=np.asarray(a, float) - np.asarray(b, float))
    observed, rng, extreme = abs(d.sum()), np.random.default_rng(seed), 0
    for size in [1000] * (n // 1000) + [n % 1000]:
        signs = rng.choice([-1.0, 1.0], size=(size, len(d)))
        extreme += int((np.abs(signs @ d) >= observed - 1e-9).sum())
    return (1 + extreme) / (n + 1)


def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values, in input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty(len(p))
    out[order] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1)
    return out


def threshold_at_far(impostor_scores, far):
    """Lowest threshold at which at most a fraction far of impostor scores is accepted
    (score >= threshold). Returns (threshold, achievable FAR resolution 1/n)."""
    s = np.sort(np.asarray(impostor_scores))[::-1]
    k = int(np.floor(far * len(s)))  # impostors allowed above the threshold
    return float(np.nextafter(s[k], np.inf)), 1 / len(s)


def threshold_for_gallery(impostor_scores, far, n):
    """Threshold t with 1 - F(t)^n = far, F being the empirical distribution of single
    impostor scores: an unknown query whose n gallery scores were independent draws
    from F would pass with probability far. n = 1 is a single-pair threshold.
    Returns (threshold, per-impostor FAR it needs, resolution 1/len)."""
    per_impostor = -math.expm1(math.log1p(-far) / n)  # 1 - (1 - far)^(1/n), stable for large n
    t, resolution = threshold_at_far(impostor_scores, per_impostor)
    return t, per_impostor, resolution
