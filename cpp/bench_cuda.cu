// CUDA top-k scorer, its exactness check against the CPU scorer, and its query latency
// next to the CPU scalar (v0_naive) and SIMD (v5_ilp) passes, at gallery sizes 10K and
// 100K, with 1,000 queries drawn from the query file. Five runs, each running every size
// and backend once, so backends are interleaved in time.
//
// The kernel scores one gallery row per thread and repeats dot4_ilp_simd's AVX2 arithmetic
// step for step, so on x86-64 its scores are bit-identical to topk_v5_ilp (rows a multiple
// of 4, dim a multiple of 16). The top k come from a stable CUB radix sort of the scores
// with row numbers as values, so ties go to the lower row, as in better().
//
// Usage: bench_cuda GALLERY.bin QUERIES.bin > bench_cuda.json
#include "scorer.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace pawmatch;

constexpr size_t K = 100, QUERIES = 1000, WARMUP = 1000, RUNS = 5, SEED = 0;
constexpr size_t SIZES[] = {10000, 100000};

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// Gallery stored transposed (dim x rows) so a warp's loads coalesce. Element i goes into
// partial sum i % 16 by fused multiply-add, as in the two eight-lane AVX2 accumulators,
// and the sixteen sums are added in the order hsum adds them.
__global__ void score(const float* __restrict__ gt, const float* __restrict__ q, uint32_t rows, uint32_t dim,
                      float* __restrict__ out) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float p[16] = {};
    for (uint32_t i = 0; i < dim; i += 16) {
#pragma unroll
        for (uint32_t c = 0; c < 16; ++c) p[c] = __fmaf_rn(q[i + c], gt[size_t(i + c) * rows + r], p[c]);
    }
    float v[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) v[l] = p[l] + p[l + 8];
    out[r] = ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}

// Device buffers are allocated once, in the constructor; the gallery stays on the GPU.
class CudaScorer {
public:
    explicit CudaScorer(const Matrix& g) : rows_(g.rows), dim_(g.dim), scores_(K), rows_out_(K) {
        if (dim_ % 16) throw std::runtime_error("dim must be a multiple of 16");
        std::vector<float> t(size_t(rows_) * dim_);
        for (size_t r = 0; r < rows_; ++r)
            for (size_t c = 0; c < dim_; ++c) t[c * rows_ + r] = g.data[r * dim_ + c];
        std::vector<uint32_t> iota(rows_);
        std::iota(iota.begin(), iota.end(), 0u);
        check(cudaStreamCreate(&stream_));
        check(cudaEventCreate(&start_));
        check(cudaEventCreate(&stop_));
        check(cudaMalloc(&gt_, t.size() * sizeof(float)));
        check(cudaMalloc(&q_, dim_ * sizeof(float)));
        check(cudaMalloc(&s_, rows_ * sizeof(float)));
        check(cudaMalloc(&sorted_, rows_ * sizeof(float)));
        check(cudaMalloc(&row_, rows_ * sizeof(uint32_t)));
        check(cudaMalloc(&row_sorted_, rows_ * sizeof(uint32_t)));
        check(cudaMemcpy(gt_, t.data(), t.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(row_, iota.data(), rows_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
        check(cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes_, s_, sorted_, row_, row_sorted_, int(rows_),
                                                        0, 32, stream_));
        check(cudaMalloc(&temp_, temp_bytes_));
    }
    CudaScorer(const CudaScorer&) = delete;
    ~CudaScorer() {
        for (void* p : {(void*)gt_, (void*)q_, (void*)s_, (void*)sorted_, (void*)row_, (void*)row_sorted_, temp_}) cudaFree(p);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaStreamDestroy(stream_);
    }

    // Copy the query in, score and sort, copy the top k out: what a host caller waits for.
    void topk(const float* query, std::span<Hit> out) {
        check(cudaMemcpyAsync(q_, query, dim_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
        launch();
        check(cudaMemcpyAsync(scores_.data(), sorted_, K * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        check(cudaMemcpyAsync(rows_out_.data(), row_sorted_, K * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_));
        check(cudaStreamSynchronize(stream_));
        for (size_t j = 0; j < out.size(); ++j) out[j] = {scores_[j], rows_out_[j]};
    }

    // GPU time of scoring and sorting alone, in microseconds; the query copy is not timed.
    double kernel_us(const float* query) {
        check(cudaMemcpyAsync(q_, query, dim_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
        check(cudaEventRecord(start_, stream_));
        launch();
        check(cudaEventRecord(stop_, stream_));
        check(cudaEventSynchronize(stop_));
        float ms;
        check(cudaEventElapsedTime(&ms, start_, stop_));
        return ms * 1000.0;
    }

private:
    void launch() {
        score<<<(rows_ + 255) / 256, 256, 0, stream_>>>(gt_, q_, rows_, dim_, s_);
        check(cudaGetLastError());
        check(cub::DeviceRadixSort::SortPairsDescending(temp_, temp_bytes_, s_, sorted_, row_, row_sorted_, int(rows_), 0,
                                                        32, stream_));
    }

    uint32_t rows_, dim_;
    float *gt_ = nullptr, *q_ = nullptr, *s_ = nullptr, *sorted_ = nullptr;
    uint32_t *row_ = nullptr, *row_sorted_ = nullptr;
    void* temp_ = nullptr;
    size_t temp_bytes_ = 0;
    cudaStream_t stream_;
    cudaEvent_t start_, stop_;
    std::vector<float> scores_;
    std::vector<uint32_t> rows_out_;
};

template <typename F>
static double wall_us(F&& f) {
    const auto start = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

struct Size {
    const Matrix& g;
    Scorer cpu;
    CudaScorer gpu;
};

struct Result {
    size_t run, gallery;
    const char* version;
    double p50_us, p99_us;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: bench_cuda GALLERY.bin QUERIES.bin\n");
        return 2;
    }
    const Matrix file = load(argv[1]), pool = load(argv[2]);
    std::mt19937 rng(SEED);

    // 1,000 queries drawn at random from the query file.
    std::vector<uint32_t> pick(pool.rows);
    std::iota(pick.begin(), pick.end(), 0u);
    std::shuffle(pick.begin(), pick.end(), rng);
    Matrix queries{uint32_t(QUERIES), pool.dim, {}};
    for (size_t i = 0; i < QUERIES; ++i) queries.data.insert(queries.data.end(), pool.row(pick[i]), pool.row(pick[i] + 1));

    // The gallery file holds 50,000 rows; the 100K gallery continues it with SYNTHETIC
    // random unit vectors from the same generator.
    Matrix big = file;
    big.rows = uint32_t(SIZES[1]);
    big.data.resize(SIZES[1] * big.dim);
    std::normal_distribution<float> normal;
    for (size_t r = file.rows; r < big.rows; ++r) {
        float* v = big.data.data() + r * big.dim;
        double norm = 0;
        for (size_t c = 0; c < big.dim; ++c) {
            v[c] = normal(rng);
            norm += double(v[c]) * v[c];
        }
        for (size_t c = 0; c < big.dim; ++c) v[c] = float(v[c] / std::sqrt(norm));
    }
    const Matrix small = head(big, SIZES[0]);
    Size sizes[] = {{small, Scorer(small, K), CudaScorer(small)}, {big, Scorer(big, K), CudaScorer(big)}};

    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    int runtime, driver;
    check(cudaRuntimeGetVersion(&runtime));
    check(cudaDriverGetVersion(&driver));
    std::printf("{\"gpu\": \"%s\", \"cuda_runtime\": %d, \"cuda_driver\": %d, \"host_compiler\": \"%s\", \"k\": %zu, "
                "\"queries\": %zu, \"query_file_rows\": %u, \"gallery_file_rows\": %u, \"seed\": %zu, "
                "\"warmup_queries\": %zu, \"runs\": %zu,\n \"exactness\": [",
                prop.name, runtime, driver, __VERSION__, K, QUERIES, pool.rows, file.rows, SEED, WARMUP, RUNS);

    // Exact means the same rows in the same order with the same score bits.
    std::vector<Hit> cpu(K), gpu(K);
    size_t failed = 0;
    for (Size& s : sizes) {
        size_t mismatched = 0;
        for (size_t i = 0; i < QUERIES; ++i) {
            s.cpu.topk_v5_ilp(queries.row(i), cpu);
            s.gpu.topk(queries.row(i), gpu);
            if (std::memcmp(cpu.data(), gpu.data(), K * sizeof(Hit)) == 0) continue;
            if (mismatched++ == 0)
                for (size_t j = 0; j < K; ++j)
                    if (std::memcmp(&cpu[j], &gpu[j], sizeof(Hit)) != 0) {
                        std::fprintf(stderr, "gallery %u query %zu rank %zu: cpu row %u score %a, gpu row %u score %a\n",
                                     s.g.rows, i, j, cpu[j].index, cpu[j].score, gpu[j].index, gpu[j].score);
                        break;
                    }
        }
        std::printf("%s\n  {\"gallery\": %u, \"queries\": %zu, \"mismatched_queries\": %zu}", &s == sizes ? "" : ",",
                    s.g.rows, QUERIES, mismatched);
        std::fflush(stdout);
        failed += mismatched;
    }
    if (failed) {
        std::printf("\n]}\n");
        std::fprintf(stderr, "FAIL: %zu queries differ from topk_v5_ilp\n", failed);
        return 1;
    }

    std::vector<Result> results;
    std::vector<double> us(QUERIES);
    float sink = 0;
    for (size_t run = 0; run < RUNS; ++run) {
        for (Size& s : sizes) {
            auto bench = [&](const char* version, auto&& query) {  // query returns its latency in microseconds
                for (size_t i = 0; i < WARMUP; ++i) query(queries.row(i % QUERIES));
                for (size_t i = 0; i < QUERIES; ++i) us[i] = query(queries.row(i));
                std::sort(us.begin(), us.end());
                auto pct = [&](double p) { return us[size_t(std::ceil(p * us.size())) - 1]; };  // nearest rank
                results.push_back({run, s.g.rows, version, pct(0.5), pct(0.99)});
                std::fprintf(stderr, "run %zu gallery %u %s p50 %.1f us\n", run, s.g.rows, version, pct(0.5));
            };
            bench("v0_naive", [&](const float* q) { return wall_us([&] { sink += topk_v0_naive(s.g, q, K)[0].score; }); });
            bench("v5_ilp", [&](const float* q) { return wall_us([&] { s.cpu.topk_v5_ilp(q, cpu); sink += cpu[0].score; }); });
            bench("cuda_with_transfer", [&](const float* q) { return wall_us([&] { s.gpu.topk(q, gpu); sink += gpu[0].score; }); });
            bench("cuda_kernel_only", [&](const float* q) { return s.gpu.kernel_us(q); });
        }
    }

    std::printf("\n ],\n \"results\": [");
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        std::printf("%s\n  {\"run\": %zu, \"version\": \"%s\", \"gallery\": %zu, \"p50_us\": %.3f, \"p99_us\": %.3f}",
                    i ? "," : "", r.run, r.version, r.gallery, r.p50_us, r.p99_us);
    }
    // Median across runs of each run's percentile.
    std::printf("\n ],\n \"summary\": [");
    for (size_t i = 0; i < results.size() / RUNS; ++i) {
        std::vector<double> p50, p99;
        for (const Result& r : results)
            if (r.gallery == results[i].gallery && std::strcmp(r.version, results[i].version) == 0)
                p50.push_back(r.p50_us), p99.push_back(r.p99_us);
        std::sort(p50.begin(), p50.end());
        std::sort(p99.begin(), p99.end());
        std::printf("%s\n  {\"version\": \"%s\", \"gallery\": %zu, \"p50_us\": %.3f, \"p99_us\": %.3f}", i ? "," : "",
                    results[i].version, results[i].gallery, p50[RUNS / 2], p99[RUNS / 2]);
    }
    std::printf("\n]}\n");
    std::fprintf(stderr, "checksum %f\n", sink);
}
