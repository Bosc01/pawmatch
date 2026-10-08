"""Retrieval and verification metrics. Tests pin these to hand-computed values."""
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
