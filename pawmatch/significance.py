"""Every significance test in the project, with Benjamini-Hochberg across all of them.

Each test compares per-query rank-1 hits of two methods on the same test queries
with a two-sided paired sign-flip permutation test at the dog level. Tests whose
inputs do not exist yet are listed as not run and are not part of the correction.

Usage: python -m pawmatch.significance CONFIG
"""
import json
import sys

import numpy as np

from pawmatch import corrupt, data, evaluate, metrics

PERMUTATIONS = 10000
ALPHA = 0.05


def tests(cfg):
    """(method a, method b) pairs, fixed in advance. Names are per-query files."""
    arm = cfg["train"].get("run_c_arm")
    return [
        # Milestone 1: frozen backbones, and the hard-negative ablation (arm b against arm a).
        ("m1_dinov2_vits14", "m1_resnet50_in1k"),
        ("m1_dinov2_vits14", "m1_clip_vitb16"),
        ("m1_resnet50_in1k", "m1_clip_vitb16"),
        ("m1_arcface_bh", "m1_arcface"),
        # Milestone 2: HNSW against exact stage 1, each rerank against stage 1, and DINOv2
        # patches against the ImageNet control under the same fusion.
        ("m2_stage1_hnsw", "m2_stage1_flat"),
        ("m2_rerank_mnn_dinov2_vits14", "m2_stage1_flat"),
        ("m2_rerank_rrf_dinov2_vits14", "m2_stage1_flat"),
        ("m2_rerank_mnn_resnet50_in1k", "m2_stage1_flat"),
        ("m2_rerank_rrf_resnet50_in1k", "m2_stage1_flat"),
        ("m2_rerank_mnn_dinov2_vits14", "m2_rerank_mnn_resnet50_in1k"),
        ("m2_rerank_rrf_dinov2_vits14", "m2_rerank_rrf_resnet50_in1k"),
        # Milestone 2: the cross-attention head against stage 1 and against the patch rerank.
        ("m2_rerank_xattn_dinov2_vits14", "m2_stage1_flat"),
        ("m2_rerank_xattn_rrf_dinov2_vits14", "m2_stage1_flat"),
        ("m2_rerank_xattn_dinov2_vits14", "m2_rerank_mnn_dinov2_vits14"),
        # Milestone 2: each corruption against clean queries for the stage-1 model; run (c)
        # against its arm on clean and on corrupted queries.
        *[("m1_dinov2_vits14", f"m2_dinov2_vits14_{c}") for c in corrupt.CORRUPTIONS],
        ("m1_augmented", f"m1_{arm}"),
        *[(f"m2_augmented_{c}", f"m2_{arm}_{c}") for c in corrupt.CORRUPTIONS],
    ]


def main(config):
    cfg = data.load_config(config)
    per_query = cfg["cache_dir"] / "per_query"
    run, not_run = [], []
    for a, b in tests(cfg):
        if not ((per_query / f"{a}.npz").exists() and (per_query / f"{b}.npz").exists()):
            not_run.append(f"{a} vs {b}")
            continue
        A, B = np.load(per_query / f"{a}.npz"), np.load(per_query / f"{b}.npz")
        assert (A["dogs"] == B["dogs"]).all(), f"{a} and {b} are not on the same queries"
        x, y = A["rank1"].astype(float), B["rank1"].astype(float)
        run.append({"a": a, "b": b, "metric": "rank-1", "n_queries": len(x), "n_dogs": len(set(A["dogs"])),
                    "mean_a": x.mean(), "mean_b": y.mean(), "difference": x.mean() - y.mean(),
                    "p": metrics.paired_permutation_p(x, y, A["dogs"], PERMUTATIONS)})
    for t, q in zip(run, metrics.benjamini_hochberg([t["p"] for t in run]) if run else []):
        t["p_bh"] = q
        t[f"significant_at_{ALPHA}"] = bool(q < ALPHA)
    out = cfg["results_dir"] / "significance.json"
    out.write_text(json.dumps({
        "dataset": cfg["dataset"]["name"], "code_commit": evaluate.commit(),
        "test": "two-sided paired sign-flip permutation test, dogs as units", "permutations": PERMUTATIONS,
        "correction": "Benjamini-Hochberg across every test that ran", "alpha": ALPHA,
        "n_tests": len(run), "tests": run, "not_run": not_run}, indent=1, default=float) + "\n")
    print(f"{out}: {len(run)} tests, {sum(t[f'significant_at_{ALPHA}'] for t in run)} significant after BH, "
          f"{len(not_run)} not run")


if __name__ == "__main__":
    main(sys.argv[1])
