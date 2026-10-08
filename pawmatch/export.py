"""Milestone 4 glue between the Python pipeline and the C++ scorer.

  python -m pawmatch.export CONFIG               fixtures from cached DINOv2 embeddings
  python -m pawmatch.export --synthetic          fixtures from random unit vectors only (CI)
  python -m pawmatch.export --report BENCH_JSON  results/m4_latency.json from bench output

Fixtures go to data/cache/m4: gallery.bin (50,000 rows), queries.bin (10,000 rows),
reference_scores.bin (every query against the first 1,000 gallery rows, computed in
float64), and manifest.json. Real rows come first: the gallery holds the test gallery
photos and then every validation and training photo, and the queries are the test
queries. The remaining rows are SYNTHETIC random unit vectors; brute-force latency
does not depend on vector values.

Format: b"PWMB", uint32 version 1, uint32 rows, uint32 dim, then rows x dim
little-endian float32, row-major.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

from pawmatch import data, evaluate

OUT = Path("data/cache/m4")
GALLERY_ROWS, QUERY_ROWS, PARITY_ROWS, DIM, SEED = 50000, 10000, 1000, 384, 0
MODEL = "dinov2_vits14"
STATS = ("p50_us", "p99_us", "p999_us", "mean_us", "heap_allocations_per_query")


def write(path, x):
    x = np.ascontiguousarray(x, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"PWMB" + struct.pack("<III", 1, *x.shape) + x.tobytes())


def _pad(real, rows, rng):
    real = real[:rows]
    v = rng.standard_normal((rows - len(real), real.shape[1])).astype(np.float32)
    return np.concatenate([real, v / np.linalg.norm(v, axis=1, keepdims=True)])


def export(gallery_real, query_real, source):
    rng = np.random.default_rng(SEED)
    gallery, queries = _pad(gallery_real, GALLERY_ROWS, rng), _pad(query_real, QUERY_ROWS, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "gallery.bin", gallery)
    write(OUT / "queries.bin", queries)
    write(OUT / "reference_scores.bin", queries.astype(np.float64) @ gallery[:PARITY_ROWS].astype(np.float64).T)
    (OUT / "manifest.json").write_text(json.dumps({
        "source": source, "dim": gallery.shape[1],
        "gallery_rows": GALLERY_ROWS, "real_gallery_rows": min(len(gallery_real), GALLERY_ROWS),
        "query_rows": QUERY_ROWS, "real_query_rows": min(len(query_real), QUERY_ROWS),
        "parity_gallery_rows": PARITY_ROWS, "synthetic_rows": "random unit vectors, seed 0"}, indent=1) + "\n")
    print(f"{OUT}: {len(gallery_real)} real gallery rows, {len(query_real)} real query rows")


def from_config(config):
    cfg = data.load_config(config)
    splits, _ = data.load_splits(cfg)
    emb = {s: np.load(cfg["cache_dir"] / MODEL / f"{s}.npz")["emb"] for s in data.SPLITS}
    g, q, _ = data.gallery_query(splits["test"], cfg["protocol"]["seed"])
    gallery = np.concatenate([emb["test"][g], emb["val_a"], emb["val_b"], emb["train"]])
    export(gallery, emb["test"][q], f"{cfg['dataset']['name']} {MODEL} embeddings")


def report(bench_json):
    bench = json.loads(Path(bench_json).read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = bench.pop("results")
    groups = {}
    for r in rows:
        groups.setdefault((r["gallery"], r["version"]), []).append(r)
    summary = [{"gallery": size, "version": version, **{k: float(np.median([r[k] for r in rs])) for k in STATS}}
               for (size, version), rs in groups.items()]
    real = manifest["real_gallery_rows"]
    out = Path("results/m4_latency.json")
    out.write_text(json.dumps({
        "hardware": evaluate.hardware(), "code_commit": evaluate.commit(), "threads": 1,
        "build": "CMake Release (-O3), no -ffast-math; NEON on arm64, AVX2 and FMA on x86-64", **bench,
        "statistic": "median across rounds of each round's percentile",
        "gallery_rows": {size: {"real": min(size, real), "synthetic": max(0, size - real)}
                         for size in sorted({r["gallery"] for r in rows})},
        "vectors": manifest, "summary": summary, "per_round": rows}, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    if sys.argv[1] == "--synthetic":
        empty = np.zeros((0, DIM), np.float32)
        export(empty, empty, "synthetic only")
    elif sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        from_config(sys.argv[1])
