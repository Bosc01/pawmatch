# CUDA top-100 scorer: exactness and latency on a Colab T4

Run on October 28, 2026 (UTC), on a Google Colab T4 GPU runtime, with the code in `cpp/bench_cuda.cu`, `scripts/run_colab.sh`, and `COLAB.md`. Every number below comes from `bench_cuda.json`, `lscpu.txt`, `nvidia-smi.txt`, and `sha256.txt` in the `results_colab.zip` that run wrote.

## Exactness

The CUDA top 100 is compared with `topk_v5_ilp`, the AVX2 pass of the existing C++ scorer, on the same machine. Exact means the same rows in the same order with bit-identical scores: a `memcmp` of all 100 (score, row) pairs.

| Gallery rows | Queries | Queries that differ from `v5_ilp` |
|---:|---:|---:|
| 10,000 | 1,000 | 0 |
| 100,000 | 1,000 | 0 |

The kernel repeats the arithmetic of `dot4_ilp_simd` on AVX2 step for step: element i goes into partial sum i mod 16 by fused multiply-add, and the sixteen sums are added in the order `hsum` adds them. The top 100 come from a stable CUB radix sort of the scores with row numbers as values, so ties go to the lower row, as in `better()`. This exactness holds against the x86-64 AVX2 build. The arm64 NEON build adds its partial sums in a different order, so it is not expected to match bit for bit.

## Latency per query, median of 5 runs

| Gallery rows | Version | p50 (µs) | p99 (µs) |
|---:|---|---:|---:|
| 10,000 | CPU scalar (`v0_naive`) | 4,716.7 | 5,499.1 |
| 10,000 | CPU AVX2 (`v5_ilp`) | 683.8 | 929.8 |
| 10,000 | CUDA, with transfers | 197.7 | 221.0 |
| 10,000 | CUDA, kernel only | 171.1 | 181.9 |
| 100,000 | CPU scalar (`v0_naive`) | 47,587.6 | 53,777.0 |
| 100,000 | CPU AVX2 (`v5_ilp`) | 13,385.6 | 17,862.2 |
| 100,000 | CUDA, with transfers | 827.8 | 854.2 |
| 100,000 | CUDA, kernel only | 806.9 | 822.3 |

The CPU timings (`v0_naive` and `v5_ilp`) come from a shared Google Colab VM with 2 vCPUs, not dedicated hardware. Other tenants on the same host can slow them, so they are not comparable with the Apple M5 timings in `results/m4_latency.json`.

At 10,000 rows, CUDA with transfers has a p50 3.5 times lower than the AVX2 pass (197.7 against 683.8 µs). At 100,000 rows the ratio is 16.2 (827.8 against 13,385.6 µs). The split of the kernel-only time between the scoring kernel and the sort was not measured.

## Hardware and software

| | |
|---|---|
| GPU | Tesla T4, 15,360 MiB, driver 580.82.07 |
| CUDA | runtime 13.0, nvcc 13.0.88, built with `CMAKE_CUDA_ARCHITECTURES=native` |
| CPU | Intel(R) Xeon(R) CPU @ 2.00GHz, family 6, model 85, stepping 3; 2 vCPUs (1 core, 2 threads per core); KVM guest; L3 38.5 MiB |
| Host build | g++ 13.3.0 (Ubuntu 24.04), CMake 3.31.10, Release (`-O3`), AVX2 and FMA, no `-ffast-math` |
| Runtime | Google Colab, T4 GPU runtime |

## What was measured

- **Galleries.** 10K is the first 10,000 rows of `data/cache/m4/gallery.bin`, the same 10K gallery as `results/m4_latency.json`: 7,013 real DogFaceNet DINOv2 embeddings and 2,987 SYNTHETIC random unit vectors. 100K is the whole 50,000-row file followed by 50,000 more SYNTHETIC random unit vectors, generated inside `bench_cuda` (`std::mt19937`, seed 0): 7,013 real and 92,987 synthetic rows.
- **Queries.** 1,000 rows drawn at random (`std::shuffle`, `std::mt19937`, seed 0) from the 10,000 rows of `data/cache/m4/queries.bin`. The same 1,000 queries are used for the exactness check and for timing.
- **Settings.** Top 100, one query at a time, one CPU thread.
- **Versions.**
  - CPU scalar is `v0_naive`: scalar dot products into a score buffer, then `partial_sort`, allocating per query.
  - CPU AVX2 is `v5_ilp`.
  - CUDA, with transfers, is host wall time for one query. It covers copying the query to the GPU, the scoring kernel, the CUB sort, copying the top 100 scores and rows back, and a stream synchronize. The gallery stays on the GPU, and host buffers are pageable memory.
  - CUDA, kernel only, is CUDA event time around the scoring kernel and the sort. The query and result copies are not timed.
- **Statistic.** For each run, gallery size, and version: 1,000 warm-up queries, then 1,000 timed queries, with p50 and p99 by nearest rank. Each of the 5 runs measures every size and version once, so versions are interleaved in time. The tables report the median across the 5 runs of each run's percentile.

## Raw numbers, every run

| Run | Gallery rows | Version | p50 (µs) | p99 (µs) |
|---:|---:|---|---:|---:|
| 0 | 10,000 | `v0_naive` | 4763.140 | 5577.724 |
| 0 | 10,000 | `v5_ilp` | 706.071 | 992.244 |
| 0 | 10,000 | `cuda_with_transfer` | 203.503 | 220.955 |
| 0 | 10,000 | `cuda_kernel_only` | 171.648 | 182.272 |
| 0 | 100,000 | `v0_naive` | 47665.666 | 53385.615 |
| 0 | 100,000 | `v5_ilp` | 13385.613 | 17862.240 |
| 0 | 100,000 | `cuda_with_transfer` | 828.176 | 853.142 |
| 0 | 100,000 | `cuda_kernel_only` | 809.280 | 822.528 |
| 1 | 10,000 | `v0_naive` | 4761.839 | 5490.267 |
| 1 | 10,000 | `v5_ilp` | 712.410 | 929.810 |
| 1 | 10,000 | `cuda_with_transfer` | 206.809 | 227.014 |
| 1 | 10,000 | `cuda_kernel_only` | 170.720 | 180.768 |
| 1 | 100,000 | `v0_naive` | 47701.268 | 54038.113 |
| 1 | 100,000 | `v5_ilp` | 13494.310 | 17993.419 |
| 1 | 100,000 | `cuda_with_transfer` | 827.819 | 857.188 |
| 1 | 100,000 | `cuda_kernel_only` | 806.880 | 819.200 |
| 2 | 10,000 | `v0_naive` | 4707.115 | 5267.555 |
| 2 | 10,000 | `v5_ilp` | 683.837 | 926.761 |
| 2 | 10,000 | `cuda_with_transfer` | 197.685 | 212.898 |
| 2 | 10,000 | `cuda_kernel_only` | 171.136 | 181.856 |
| 2 | 100,000 | `v0_naive` | 47587.608 | 53777.049 |
| 2 | 100,000 | `v5_ilp` | 13448.214 | 18884.407 |
| 2 | 100,000 | `cuda_with_transfer` | 826.884 | 854.246 |
| 2 | 100,000 | `cuda_kernel_only` | 806.080 | 822.272 |
| 3 | 10,000 | `v0_naive` | 4702.024 | 5499.053 |
| 3 | 10,000 | `v5_ilp` | 664.775 | 976.912 |
| 3 | 10,000 | `cuda_with_transfer` | 192.810 | 247.965 |
| 3 | 10,000 | `cuda_kernel_only` | 171.136 | 183.072 |
| 3 | 100,000 | `v0_naive` | 47575.119 | 54036.301 |
| 3 | 100,000 | `v5_ilp` | 13327.026 | 17507.598 |
| 3 | 100,000 | `cuda_with_transfer` | 827.507 | 866.460 |
| 3 | 100,000 | `cuda_kernel_only` | 803.296 | 819.648 |
| 4 | 10,000 | `v0_naive` | 4716.716 | 6204.687 |
| 4 | 10,000 | `v5_ilp` | 660.129 | 929.685 |
| 4 | 10,000 | `cuda_with_transfer` | 191.021 | 211.494 |
| 4 | 10,000 | `cuda_kernel_only` | 171.104 | 180.512 |
| 4 | 100,000 | `v0_naive` | 47088.012 | 52384.341 |
| 4 | 100,000 | `v5_ilp` | 13088.548 | 17579.435 |
| 4 | 100,000 | `cuda_with_transfer` | 828.958 | 852.306 |
| 4 | 100,000 | `cuda_kernel_only` | 806.912 | 823.296 |

Run header from `bench_cuda.json`: GPU "Tesla T4", CUDA runtime and driver API version 13000, host compiler 13.3.0, k 100, 1,000 queries from a 10,000-row query file, a 50,000-row gallery file, seed 0, 1,000 warm-up queries, 5 runs.

## Provenance

SHA-256 hashes written on the Colab machine (`sha256.txt`). They match the files in this repository at the time this file was written.

```
2ca3dcf0ddea4428b0d1304548535e33bb1c4efe76afb4b24e2c144538926b1b data/cache/m4/gallery.bin
7349643728579b4ae480afd14471a82ef87884e30b6cec7165679f416870f9a0 data/cache/m4/queries.bin
4f07e3e15679b10679b6035cefe7966d0c12542d58acf841ededd4dd9ffb11f7 cpp/bench_cuda.cu
83ad2a9caf9eae9aaaa9c547361aca5e89100b85ebca89c3e69025bfbfecb17f cpp/scorer.cpp
c082848c45a9cd4af0acd9a4583b251830cef1736a1919f1e108cec10576a107 cpp/scorer.h
b9b8a364eba5c820130dc16ef81be22e235f04231c4d6097fa030944072e77e2 cpp/CMakeLists.txt
657ca424d10b4e6e7eb76f75f4de6c82d585a1798c5a429cd687eb5767d29cc8 Makefile
```

## Commands

On the Mac, from the repo root. The zip holds the working tree as it was, uncommitted CUDA files included. The browser upload accepted at most 10 MB per file, so the fixtures went up in 7,000,000-byte parts:

```bash
cd ~/pawmatch && rm -rf ~/Desktop/pawmatch-colab && mkdir ~/Desktop/pawmatch-colab && git ls-files -co --exclude-standard | zip -q ~/Desktop/pawmatch-colab/pawmatch.zip -@ && cp data/cache/m4/gallery.bin data/cache/m4/queries.bin ~/Desktop/pawmatch-colab/
cd ~/Desktop/pawmatch-colab && split -b 7000000 gallery.bin gallery.bin.part. && split -b 7000000 queries.bin queries.bin.part.
mkdir -p ~/pawmatch/data/cache/m4/colab_upload && mv ~/Desktop/pawmatch-colab/pawmatch.zip ~/Desktop/pawmatch-colab/*.part.* ~/pawmatch/data/cache/m4/colab_upload/
```

In Colab, with the runtime set to T4 GPU under Runtime > Change runtime type, the 15 files went to `/content` through the Files panel. Then these cells ran:

```
!nvidia-smi
!cd /content && ls -l *.part.* pawmatch.zip && cat gallery.bin.part.* > gallery.bin && cat queries.bin.part.* > queries.bin && sha256sum gallery.bin queries.bin
!cd /content && rm -rf pawmatch && unzip -q pawmatch.zip -d pawmatch && mkdir -p pawmatch/data/cache/m4 && mv gallery.bin queries.bin pawmatch/data/cache/m4/ && bash pawmatch/scripts/run_colab.sh
from google.colab import files; files.download('/content/pawmatch/results_colab.zip')
```

The joined `gallery.bin` and `queries.bin` hashed to the values in Provenance before the run. `scripts/run_colab.sh` runs `make bench-cuda`, which is:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build cpp/build -j --target bench_cuda
cpp/build/bench_cuda data/cache/m4/gallery.bin data/cache/m4/queries.bin > data/cache/m4/bench_cuda.json
```

To run it by hand without split parts, follow `COLAB.md`.
