# PawMatch

<!-- problem:start -->
US shelters and rescues took in an estimated 5.8 million dogs and cats in 2025 and returned 638,000 of them to their owners (Shelter Animals Count, 2025 Annual Data Report). In a 2009 study of 53 shelters, microchipped stray dogs went home 52.2% of the time, against 21.9% for all stray dogs (Lord et al., J Am Vet Med Assoc 2009). Sources and caveats are in `results/external.json`.
<!-- problem:end -->

Open-set dog re-identification for lost-pet reunification. Work in progress: the pipeline runs, and no PetFace results exist yet.

## Results

<!-- m1:start -->
No PetFace results yet.
<!-- m1:end -->

## C++ serving

<!-- m4:start -->
Apple M5, 24 GB RAM, 1 thread, top 100, one query at a time. Each figure is the median across 3 rounds of that round's percentile over 10,000 queries. Gallery 1,000 rows: 1,000 real and 0 synthetic, 10,000 rows: 7,013 real and 2,987 synthetic, 50,000 rows: 7,013 real and 42,987 synthetic; synthetic rows are random unit vectors, and brute-force latency does not depend on vector values. Real rows are dogfacenet dinov2_vits14 embeddings.

| Gallery | Version | p50 (us) | p99 (us) | p99.9 (us) | Heap allocations per query |
|---|---|---|---|---|---|
| 1,000 | v0_naive | 216.0 | 249.4 | 329.7 | 3 |
| 1,000 | v1_simd | 37.2 | 39.8 | 68.6 | 3 |
| 1,000 | v2_prealloc | 40.7 | 67.1 | 99.8 | 0 |
| 1,000 | v3_heap | 50.8 | 66.1 | 114.6 | 0 |
| 1,000 | v4_blocked | 48.5 | 54.6 | 84.9 | 0 |
| 1,000 | v5_ilp | 43.5 | 62.4 | 476.0 | 0 |
| 10,000 | v0_naive | 1667.3 | 3238.5 | 4833.6 | 3 |
| 10,000 | v1_simd | 342.4 | 403.7 | 656.5 | 3 |
| 10,000 | v2_prealloc | 372.6 | 445.5 | 480.0 | 0 |
| 10,000 | v3_heap | 402.4 | 680.5 | 953.7 | 0 |
| 10,000 | v4_blocked | 371.2 | 742.8 | 1066.8 | 0 |
| 10,000 | v5_ilp | 323.5 | 341.4 | 350.1 | 0 |
| 50,000 | v0_naive | 8568.5 | 13105.5 | 21925.8 | 3 |
| 50,000 | v1_simd | 1855.7 | 3980.8 | 9310.7 | 3 |
| 50,000 | v2_prealloc | 1959.5 | 4126.6 | 8707.1 | 0 |
| 50,000 | v3_heap | 1877.9 | 3535.5 | 8137.2 | 0 |
| 50,000 | v4_blocked | 2028.5 | 4257.6 | 10708.6 | 0 |
| 50,000 | v5_ilp | 1802.5 | 3171.9 | 6640.3 | 0 |
<!-- m4:end -->

## Note on the vecstore ablation

On DogFaceNet development data, vecstore's brute-force search measured faster than FAISS `IndexFlatIP`, with both pinned to one thread. vecstore calls NumPy, and this NumPy build runs its matrix-vector product in Apple Accelerate, while FAISS uses its own SIMD loop. The win belongs to Accelerate, not to vecstore's code, and it is not claimed as a project result.

`make readme` generates the tables above from `results/*.json`. DogFaceNet runs used to build and test the pipeline are written to `results/dev/` and never appear here; `tests/test_report.py` enforces that.

## Running

```
make data        # build, check, and freeze the identity-disjoint split
make embed       # cache embeddings, corrupted test sets, and patch features
make eval        # Milestone 1 metrics with bootstrap intervals
make train       # fine-tune (smoke steps only outside PetFace)
make rerank-head # cross-attention rerank head (20-step cap outside PetFace)
make retrieve    # Milestone 2: stage 1, rerank, open set, corruptions, significance
make detect      # face detector timing and detection failure on the corruption set
make serve       # POST /match demo service (PORT=8000 by default)
make export      # flat binary fixtures for the C++ scorer
make cpp-test    # GoogleTest parity, top-k, and allocation tests
make bench       # C++ latency benchmark
make test        # Python tests
```

Every Python target reads `CONFIG`, which defaults to `configs/dogfacenet.toml`. Use `CONFIG=configs/petface.toml` for PetFace.

## Plan, data, and decisions

See [BRIEF.md](BRIEF.md) for the plan and every decision, [decisions.md](decisions.md) for the reasoning, and [DATA.md](DATA.md) for data access and protocols.
