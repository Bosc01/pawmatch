#!/usr/bin/env bash
# The CUDA benchmark (cpp/bench_cuda.cu) on a Google Colab GPU runtime, with its JSON and
# everything printed along the way packed into results_colab.zip. COLAB.md has the cells.
#
# Usage, on the Colab machine, with data/cache/m4/gallery.bin and queries.bin in place:
#   bash scripts/run_colab.sh
# The zip is written even when the build or the exactness check fails, so the log comes
# back either way, and the script exits with the benchmark's status.
set -uo pipefail
cd "$(dirname "$0")/.."

out=results_colab
rm -rf "$out" "$out.zip"
mkdir -p "$out"

bench() {
    set -e
    for f in data/cache/m4/gallery.bin data/cache/m4/queries.bin; do
        [ -s "$f" ] || { echo "missing $f" >&2; return 2; }
    done
    grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo || { echo "this CPU lacks AVX2 or FMA" >&2; return 2; }
    command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
    command -v nvcc >/dev/null || { echo "no nvcc: switch the runtime to a GPU" >&2; return 2; }
    export CUDACXX=$(command -v nvcc)
    # CMAKE_CUDA_ARCHITECTURES=native needs CMake 3.24 or later.
    have=$(cmake --version 2>/dev/null | awk 'NR == 1 {print $3}') || true
    if ! printf '%s\n' 3.24 "${have:-0}" | sort -V -C; then
        pip install -q 'cmake>=3.24'
        hash -r
    fi
    cmake --version | sed -n 1p
    nvcc --version
    g++ --version | sed -n 1p
    make bench-cuda
}

bench 2>&1 | tee "$out/run.log"
status=${PIPESTATUS[0]}

cp data/cache/m4/bench_cuda.json "$out/" 2>/dev/null
nvidia-smi > "$out/nvidia-smi.txt" 2>&1
lscpu > "$out/lscpu.txt" 2>&1
sha256sum data/cache/m4/gallery.bin data/cache/m4/queries.bin cpp/bench_cuda.cu cpp/scorer.cpp cpp/scorer.h \
    cpp/CMakeLists.txt Makefile > "$out/sha256.txt" 2>&1
cat > "$out/NOTE.txt" <<EOF
The CPU timings (v0_naive and v5_ilp) come from a shared Google Colab VM with $(nproc 2>&1) vCPUs, not
dedicated hardware. Other tenants on the same host can slow them, so they are not comparable with the
Apple M5 timings in results/m4_latency.json. The CPU model is in lscpu.txt.
Benchmark exit status: $status
EOF
python3 -m zipfile -c "$out.zip" "$out"
echo "wrote $PWD/$out.zip, benchmark exit status $status"
exit "$status"
