"""Milestone 2: two-stage retrieval, open-set thresholds, and robustness to SYNTHETIC
kennel-photo corruptions, on the test split with the Milestone 1 protocol.

Usage: python -m pawmatch.retrieve CONFIG [JOB ...]
Jobs m2_ann, m2_rerank, m2_open_set, and m2_corruption (default: all) each write
<job>.json to the results directory, and per-query rank-1 hits to the cache for
pawmatch.significance. m2_ann times searches and refuses to run unless NumPy's BLAS
is pinned to one thread through the environment (the Makefile does this).
Every value below was fixed before any run; none is tuned.
"""
import json
import os
import sys
import time

import faiss
import numpy as np

from pawmatch import corrupt, data, evaluate, metrics

STAGE1 = "dinov2_vits14"  # stage-1 model; its patch tokens drive the main rerank
CONTROL = "resnet50_in1k"  # control arm: the same rerank on ImageNet feature-map cells
MODELS = ("resnet50_in1k", "clip_vitb16", "dinov2_vits14", "arcface", "arcface_bh", "augmented")
TOP = evaluate.STAGE1_TOP
RRF_K = 60  # reciprocal rank fusion constant from Cormack et al., SIGIR 2009
HNSW_M, HNSW_EF_CONSTRUCTION = 32, 40
EF_SEARCH = (16, 32, 64, 128, 256)
EF_TESTED = 64  # the one HNSW operating point entered in the significance tests
FARS = (0.01, 0.001)
CHUNK = 1024
VECSTORE_SOURCE = "github.com/Bosc01/vecstore, branch packaging, commit 4c4096c"

try:  # pure NumPy brute force and HNSW, the third stage-1 ablation (decisions.md)
    import vecstore
except ImportError:
    vecstore = None


def _load(cfg, tag, name):
    path = cfg["cache_dir"] / tag / f"{name}.npz"
    return np.load(path) if path.exists() else None


def _true_rank(ids, q_dogs, g_dogs):
    """1-based position of each query's true gallery photo in its result list, 0 if absent."""
    hit = (ids >= 0) & (g_dogs[ids] == q_dogs[:, None])
    return np.where(hit.any(1), hit.argmax(1) + 1, 0)


def _timed(search, Q):
    """One query at a time, as a serving path would issue them. search(x) returns ids."""
    ids, ms = [], []
    for x in Q:
        start = time.perf_counter()
        ids.append(search(x))
        ms.append(1000 * (time.perf_counter() - start))
    return np.array(ids), {"p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99))}


def _recall(ids, exact):
    return float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ids, exact)]))


def _vecstore(db, Q, exact):
    """vecstore brute force and HNSW at the FAISS HNSW settings, timed the same way,
    with NumPy's BLAS pinned to one thread like FAISS."""
    if vecstore is None:
        return {"status": "not installed", "source": VECSTORE_SOURCE}
    store = vecstore.VectorStore(db.shape[1])
    store.add_batch(enumerate(db))
    ids, latency = _timed(lambda x: [i for i, _ in store.search(x, 10)], Q)
    brute = {"recall_at_10": _recall(ids, exact), **latency}
    start = time.perf_counter()
    index = vecstore.HNSWIndex(db.shape[1], M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION)
    for i, v in enumerate(db):
        index.add(i, v)
    build = time.perf_counter() - start
    sweep = []
    for ef in EF_SEARCH:
        ids, latency = _timed(lambda x: [i for i, _ in index.search(x, 10, ef_search=ef)], Q)
        sweep.append({"ef_search": ef, "recall_at_10": _recall(ids, exact), **latency})
    return {"status": "run", "source": VECSTORE_SOURCE, "brute_force": brute,
            "hnsw": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION,
                     "flags": "defaults: heuristic_select False, layer0_full_links True",
                     "build_seconds": build, "sweep": sweep}}


def _hnsw(vectors, ef_search):
    index = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(vectors)
    index.hnsw.efSearch = ef_search
    return index


def ann(cfg, splits, seed, n_boot):
    """Exact (IndexFlatIP) and HNSW stage-1 search with FAISS on one thread, and the
    vecstore ablation. ANN recall@10 is measured against exact FAISS search over every
    photo that is not a test query, and needs no labels. Identification recall uses
    the test gallery."""
    pins = {k: os.environ.get(k) for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    if "1" not in (pins["VECLIB_MAXIMUM_THREADS"], pins["OPENBLAS_NUM_THREADS"]):
        sys.exit("m2_ann times searches: set VECLIB_MAXIMUM_THREADS=1 (Accelerate) or OPENBLAS_NUM_THREADS=1")
    faiss.omp_set_num_threads(1)
    parts = [_load(cfg, STAGE1, s) for s in data.SPLITS]
    if any(p is None for p in parts):
        sys.exit("stage 1 needs all four splits embedded; run make embed")
    g, q, _ = data.gallery_query(splits["test"], seed)
    test, dogs = parts[3]["emb"], parts[3]["dogs"]
    db = np.ascontiguousarray(np.concatenate([p["emb"] for p in parts[:3]] + [test[g]]))
    Q, G = np.ascontiguousarray(test[q]), np.ascontiguousarray(test[g])
    qd, gd = dogs[q], dogs[g]

    flat = faiss.IndexFlatIP(db.shape[1])
    flat.add(db)
    exact, flat_latency = _timed(lambda x: flat.search(x[None], 10)[1][0], Q)
    start = time.perf_counter()
    hnsw = _hnsw(db, EF_SEARCH[0])
    build = time.perf_counter() - start
    sweep = []
    for ef in EF_SEARCH:
        hnsw.hnsw.efSearch = ef
        ids, latency = _timed(lambda x: hnsw.search(x[None], 10)[1][0], Q)
        sweep.append({"ef_search": ef, "recall_at_10": _recall(ids, exact), **latency})

    top = min(TOP, len(G))
    flat_g = faiss.IndexFlatIP(G.shape[1])
    flat_g.add(G)
    identification = {}
    for name, index in (("flat", flat_g), (f"hnsw_ef{EF_TESTED}", _hnsw(G, EF_TESTED))):
        rank = _true_rank(index.search(Q, top)[1], qd, gd)
        identification[name] = {
            "recall_at_1": evaluate._score(rank == 1, qd, n_boot, seed),
            f"recall_at_{top}": evaluate._score(rank > 0, qd, n_boot, seed),
        }
        evaluate.save_per_query(cfg, f"m2_stage1_{name.split('_')[0]}", rank1=rank == 1, dogs=qd)
    return {
        "model": STAGE1, "hardware": evaluate.hardware(), "faiss": faiss.__version__, "threads": 1,
        "thread_pinning": {"faiss_omp_threads": 1, **pins},
        "ann": {"database_size": len(db), "n_queries": len(Q), "k": 10, "flat": flat_latency,
                "hnsw": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION,
                         "build_seconds": build, "sweep": sweep}},
        "vecstore": _vecstore(db, Q, exact),
        "identification": {"gallery_size": len(G), "n_queries": len(Q), **identification},
    }


def mnn_scores(a, b):
    """Sum of similarities over mutual nearest-neighbour local-feature pairs.
    a: (p, d) query features; b: (c, p', d) candidate features; all L2-normalized."""
    s = b @ a.T  # (c, p', p)
    fwd = s.argmax(1)  # (c, p): nearest candidate feature for each query feature
    bwd = s.argmax(2)  # (c, p'): nearest query feature for each candidate feature
    mutual = np.take_along_axis(bwd, fwd, 1) == np.arange(a.shape[0])
    best = np.take_along_axis(s, fwd[:, None, :], 1)[:, 0]
    return (best * mutual).sum(1)


def _rrf(scores):
    """Reciprocal rank fusion of the stage-1 order (columns) and the stage-2 order."""
    stage1 = np.arange(1, scores.shape[1] + 1)
    stage2 = np.argsort(np.argsort(-scores, 1, kind="stable"), 1) + 1
    return 1 / (RRF_K + stage1) + 1 / (RRF_K + stage2)


def rerank(cfg, splits, sha, seed, n_boot):
    """Rerank the stage-1 top 100 by mutual nearest-neighbour local-feature matching,
    with DINOv2 patch tokens and, as the control arm, ImageNet ResNet-50 feature-map
    cells; and, when its scores exist, by the cross-attention head (pawmatch.xattn).
    Fusions: stage-2 score alone, and RRF with the stage-1 order. A true match outside
    the top 100 keeps its stage-1 rank. Ties count against the true match."""
    items = splits["test"]
    dogs = np.array([d for _, d in items])
    g, q, _ = data.gallery_query(items, seed)
    g = np.asarray(g)
    qd, gd = dogs[q], dogs[g]
    emb = _load(cfg, STAGE1, "test")["emb"]
    stage1_rank = evaluate._ranks(emb[q], qd, emb[g], gd)
    top = evaluate.stage1_top(emb[q], emb[g], TOP)
    top_n = top.shape[1]
    hit = gd[top] == qd[:, None]
    in_top, pos = hit.any(1), hit.argmax(1)

    def summary(rank):
        return {m: evaluate._score(v, qd, n_boot, seed)
                for m, v in (("rank1", rank == 1), ("rank5", rank <= 5), ("mrr", 1 / rank))}

    out = {"stage1_model": STAGE1, "control_model": CONTROL, "top": top_n, "rrf_k": RRF_K,
           "fusion_selection": ("no fusion weight is tuned: 'mnn' ranks by the stage-2 score alone, and 'rrf' "
                                "weights the stage-1 and stage-2 ranks equally with k = 60 from Cormack et al. "
                                "(SIGIR 2009); both were fixed in code before the first run and no split was "
                                "used to choose them"),
           "gallery_size": len(g), "n_queries": len(q),
           "stage1_recall_at_top": evaluate._score(in_top, qd, n_boot, seed),
           "stage1": summary(stage1_rank)}
    def apply(name, score):
        t = score[np.arange(len(q)), pos]
        rank = np.where(in_top, (score >= t[:, None]).sum(1), stage1_rank)
        out[name] = summary(rank)
        evaluate.save_per_query(cfg, f"m2_{name}", rank1=rank == 1, dogs=qd)

    for feat in (STAGE1, CONTROL):
        P = np.load(cfg["cache_dir"] / feat / "test_patches.npy", mmap_mode="r")
        mnn = np.stack([mnn_scores(np.asarray(P[q[i]], np.float32), np.asarray(P[g[top[i]]], np.float32))
                        for i in range(len(q))])
        apply(f"rerank_mnn_{feat}", mnn)
        apply(f"rerank_rrf_{feat}", _rrf(mnn))
    xattn = cfg["cache_dir"] / "xattn_scores.npz"
    if xattn.exists():
        z = np.load(xattn)
        if str(z["sha256"]) != sha or not np.array_equal(z["top"], top):
            sys.exit(f"{xattn} does not match the current stage-1 candidates; rerun make rerank-head")
        apply(f"rerank_xattn_{STAGE1}", z["scores"])
        apply(f"rerank_xattn_rrf_{STAGE1}", _rrf(z["scores"]))
    return out


def _pairs(emb, dogs, thresholds=None):
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


def _best(emb, items, seed, frac):
    """Open-set protocol: top gallery score and its correctness for every query, and
    the gallery size."""
    dogs = np.array([d for _, d in items])
    g, q, unknown = data.gallery_query(items, seed, frac)
    best = np.concatenate([emb[q[i:i + CHUNK]] @ emb[g].T for i in range(0, len(q), CHUNK)])
    correct = dogs[g][best.argmax(1)] == dogs[q]
    return best.max(1), correct, ~np.isin(dogs[q], list(unknown)), dogs[q], len(g)


def open_set(cfg, splits, seed, frac, n_boot):
    """Thresholds come from val_a and are applied unchanged to test; the achieved test
    FAR is reported next to every target.

    Both thresholds use F, the distribution of single impostor scores, estimated from
    every pair of val_a photos of different dogs. Verification (TAR, all pairs of
    distinct test photos) accepts a pair at or above t with F(t) = 1 - FAR.
    Identification (DIR) accepts a query's top match at or above t with
    1 - F(t)^N = FAR, N being the test gallery size: the chance that an unknown dog's
    best score passes if its N gallery scores were independent draws from F. DIR counts
    a known query only if its top match is the right dog and passes. A target that needs
    a per-impostor rate below the val_a resolution (one pair in n) is flagged."""
    out = {}
    for tag in MODELS:
        va, te = _load(cfg, tag, "val_a"), _load(cfg, tag, "test")
        if va is None or te is None:
            continue
        _, _, imp = _pairs(va["emb"], va["dogs"])
        best, correct, known, qd, n_gallery = _best(te["emb"], splits["test"], seed, frac)
        tv = [metrics.threshold_for_gallery(imp, f, 1) for f in FARS]
        ti = [metrics.threshold_for_gallery(imp, f, n_gallery) for f in FARS]
        gen, gen_dogs, test_far_v = _pairs(te["emb"], te["dogs"], [t for t, _, _ in tv])
        res = {}
        for k, far in enumerate(FARS):
            (t_v, _, res_v), (t_i, per_i, res_i) = tv[k], ti[k]
            res[f"far_{far}"] = {
                "verification": {
                    "target_far": far, "test_far": float(test_far_v[k]), "threshold": t_v,
                    "val_a_resolution": res_v, "below_resolution": far < res_v,
                    "tar": evaluate._score(gen >= t_v, gen_dogs, n_boot, seed)},
                "identification": {
                    "target_far": far, "test_far": float((best[~known] >= t_i).mean()), "threshold": t_i,
                    "gallery_size": n_gallery, "per_impostor_far": per_i,
                    "val_a_resolution": res_i, "below_resolution": per_i < res_i,
                    "dir": evaluate._score((correct & (best >= t_i))[known], qd[known], n_boot, seed)},
            }
        out[tag] = {"n_val_a_impostor_pairs": len(imp), "n_test_genuine_pairs": len(gen),
                    "n_test_known_queries": int(known.sum()), "n_test_unknown_queries": int((~known).sum()), **res}
    return {"unknown_frac": frac, "far_targets": list(FARS),
            "calibration": "val_a impostor score distribution F; identification threshold solves 1 - F(t)^N = FAR",
            "models": out}


def corruption(cfg, splits, seed, n_boot):
    """Closed-set rank-1 with SYNTHETIC corruptions applied to the queries only."""
    items = splits["test"]
    dogs = np.array([d for _, d in items])
    g, q, _ = data.gallery_query(items, seed)
    qd, gd = dogs[q], dogs[g]
    out = {}
    for tag in MODELS:
        clean = _load(cfg, tag, "test")
        if clean is None:
            continue
        G = clean["emb"][g]
        base = evaluate._ranks(clean["emb"][q], qd, G, gd) == 1
        res = {"clean_rank1": evaluate._score(base, qd, n_boot, seed)}
        for c in corrupt.CORRUPTIONS:
            z = _load(cfg, tag, f"test_{c}")
            if z is None:
                continue
            hit = evaluate._ranks(z["emb"][q], qd, G, gd) == 1
            res[c] = {"rank1": evaluate._score(hit, qd, n_boot, seed),
                      "drop": evaluate._score(base.astype(float) - hit, qd, n_boot, seed)}
            evaluate.save_per_query(cfg, f"m2_{tag}_{c}", rank1=hit, dogs=qd)
        out[tag] = res
    params = {"blur_length_px": corrupt.BLUR_LENGTH, "low_light_scale_gamma_noise": corrupt.LOW_LIGHT,
              "jpeg_quality": corrupt.JPEG_QUALITY, "crop_area": corrupt.CROP_AREA}
    return {"synthetic": True, "applied_to": "queries only", "parameters": params, "models": out}


def main(config, *names):
    cfg = data.load_config(config)
    splits, sha = data.load_splits(cfg)
    p = cfg["protocol"]
    common = {"dataset": cfg["dataset"]["name"], "split": "test", "split_sha256": sha,
              "protocol_seed": p["seed"], "bootstrap_resamples": p["bootstrap"], "code_commit": evaluate.commit()}
    jobs = {"m2_ann": lambda: ann(cfg, splits, p["seed"], p["bootstrap"]),
            "m2_rerank": lambda: rerank(cfg, splits, sha, p["seed"], p["bootstrap"]),
            "m2_open_set": lambda: open_set(cfg, splits, p["seed"], p["unknown_frac"], p["bootstrap"]),
            "m2_corruption": lambda: corruption(cfg, splits, p["seed"], p["bootstrap"])}
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    for name in names or jobs:
        job = jobs[name]
        start = time.perf_counter()
        out = cfg["results_dir"] / f"{name}.json"
        out.write_text(json.dumps({**common, **job()}, indent=1, default=float) + "\n")
        print(f"{out} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main(*sys.argv[1:])
