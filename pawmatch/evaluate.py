"""Milestone 1 evaluation of cached test-split embeddings.

Usage: python -m pawmatch.evaluate CONFIG TAG
Writes <results_dir>/m1_<TAG>.json. Every interval is a 95% bootstrap over query dogs.
"""
import json
import platform
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from pawmatch import data, metrics

CHUNK = 1024
STAGE1_TOP = 100  # candidates passed to stage 2


def _score(values, groups, n_boot, seed):
    values = np.asarray(values, float)
    ci = metrics.bootstrap_ci(lambda w: np.average(values, weights=w), groups, n_boot, seed)
    return {"value": float(values.mean()), "ci95": ci}


def stage1_top(Q, G, top=STAGE1_TOP):
    """Stage-1 candidates: gallery positions of each query's top scores, best first."""
    n = min(top, len(G))
    return np.concatenate([np.argsort(-(Q[i:i + CHUNK] @ G.T), 1, kind="stable")[:, :n]
                           for i in range(0, len(Q), CHUNK)])


def _ranks(Q, q_dogs, G, g_dogs):
    return np.concatenate([metrics.match_ranks(Q[i:i + CHUNK] @ G.T, q_dogs[i:i + CHUNK], g_dogs)
                           for i in range(0, len(Q), CHUNK)])


def compute(items, emb, pairs, seed, unknown_frac, n_boot):
    dogs = np.array([d for _, d in items])
    out = {}

    g, q, _ = data.gallery_query(items, seed)
    qd = dogs[q]
    ranks = _ranks(emb[q], qd, emb[g], dogs[g])
    out["closed_set"] = {
        "n_gallery": len(g), "n_queries": len(q), "n_query_dogs": len(set(qd)),
        "rank1": _score(ranks == 1, qd, n_boot, seed),
        "rank5": _score(ranks <= 5, qd, n_boot, seed),
        "mrr": _score(1 / ranks, qd, n_boot, seed),
    }

    g, q_open, unknown = data.gallery_query(items, seed, unknown_frac)
    assert q_open == q
    known = ~np.isin(qd, list(unknown))
    Q, G = emb[q], emb[g]
    best = np.concatenate([(Q[i:i + CHUNK] @ G.T).max(1) for i in range(0, len(Q), CHUNK)])
    open_ranks = _ranks(Q[known], qd[known], G, dogs[g])
    out["open_set"] = {
        "unknown_frac": unknown_frac, "n_gallery": len(g),
        "n_known_queries": int(known.sum()), "n_unknown_queries": int((~known).sum()),
        "rank1_known": _score(open_ranks == 1, qd[known], n_boot, seed),
        # Threshold-free: how well the top gallery similarity separates known from unknown queries.
        "detection_auroc": {
            "value": float(roc_auc_score(known, best)),
            "ci95": metrics.bootstrap_ci(lambda w: roc_auc_score(known, best, sample_weight=w), qd, n_boot, seed),
        },
    }

    # mAP needs several relevant items per query, which the one-photo gallery cannot give
    # (there AP equals reciprocal rank). So every test photo of a dog with two or more
    # photos queries all other test photos.
    ids, counts = np.unique(dogs, return_counts=True)
    rows = np.flatnonzero(np.isin(dogs, ids[counts >= 2]))
    ap = []
    for i in range(0, len(rows), CHUNK):
        r = rows[i:i + CHUNK]
        S, rel = emb[r] @ emb.T, dogs[r][:, None] == dogs[None, :]
        S[np.arange(len(r)), r], rel[np.arange(len(r)), r] = -np.inf, False
        ap.append(metrics.average_precision(S, rel))
    out["map_leave_one_out"] = {"n_queries": len(rows), **_score(np.concatenate(ap), dogs[rows], n_boot, seed)}

    i, j, y = (np.array(c) for c in zip(*pairs))
    s = (emb[i] * emb[j]).sum(1)

    def auc(clamp):
        ci = metrics.bootstrap_ci(lambda w: metrics.verification_auc(y, s, clamp, w), dogs[i], n_boot, seed)
        return {"value": float(metrics.verification_auc(y, s, clamp)), "ci95": ci}

    out["verification"] = {"n_pairs": len(y), "auc": auc(False), "auc_clamped_as_paper": auc(True)}
    return out, {"rank1": ranks == 1, "dogs": qd}


def pair_scores(emb, dogs, thresholds=None):
    """All unordered pairs of distinct photos, split into same-dog (genuine) and
    different-dog (impostor) scores. With thresholds, impostors are only counted."""
    gen, gen_dogs, imp, accepted, n_imp = [], [], [], np.zeros(len(thresholds or [])), 0
    for i in range(0, len(emb), CHUNK):
        rows = np.arange(i, min(i + CHUNK, len(emb)))
        S = emb[rows] @ emb.T
        upper = np.arange(len(emb))[None, :] > rows[:, None]
        same = dogs[rows][:, None] == dogs[None, :]
        gen.append(S[upper & same])
        gen_dogs.append(np.broadcast_to(dogs[rows][:, None], S.shape)[upper & same])
        s = S[upper & ~same]
        if thresholds is None:
            imp.append(s)
        else:
            accepted += [(s >= t).sum() for t in thresholds]
            n_imp += s.size
    gen, gen_dogs = np.concatenate(gen), np.concatenate(gen_dogs)
    return (gen, gen_dogs, np.concatenate(imp)) if thresholds is None else (gen, gen_dogs, accepted / n_imp)


def save_per_query(cfg, name, **arrays):
    """Per-query arrays for pawmatch.significance, kept in the cache (not committed)."""
    out = cfg["cache_dir"] / "per_query" / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)


def commit():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return "uncommitted changes" if git("status", "--porcelain", "pawmatch", "cpp") else git("rev-parse", "--short", "HEAD")


def hardware():
    run = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()
    if sys.platform == "darwin":
        return f"{run('sysctl', '-n', 'machdep.cpu.brand_string')}, {int(run('sysctl', '-n', 'hw.memsize')) // 2**30} GB RAM"
    try:
        gpu = run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    except FileNotFoundError:
        gpu = ""
    return f"{platform.platform()}; GPU: {gpu or 'none'}"


def main(config, tag):
    cfg = data.load_config(config)
    splits, sha = data.load_splits(cfg)
    items = splits["test"]
    z = np.load(cfg["cache_dir"] / tag / "test.npz")
    if z["paths"].tolist() != [p for p, _ in items]:
        sys.exit("embedding cache does not match the split; rerun make embed")
    p = cfg["protocol"]
    official = data.official_pairs(cfg, items)
    pairs = official or data.verification_pairs(items, p["seed"])
    result = {
        "dataset": cfg["dataset"]["name"], "model": tag, "split": "test", "split_sha256": sha,
        "protocol_seed": p["seed"], "bootstrap_resamples": p["bootstrap"],
        "verification_pairs": "official verification.csv" if official else "generated, PetFace protocol",
        "code_commit": commit(),
    }
    metrics_out, per_query = compute(items, z["emb"], pairs, p["seed"], p["unknown_frac"], p["bootstrap"])
    result.update(metrics_out)
    save_per_query(cfg, f"m1_{tag}", **per_query)
    out = cfg["results_dir"] / f"m1_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    c = result["closed_set"]
    print(f"{out}: rank1 {c['rank1']['value']:.4f} rank5 {c['rank5']['value']:.4f} "
          f"mAP {result['map_leave_one_out']['value']:.4f} AUC {result['verification']['auc']['value']:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
