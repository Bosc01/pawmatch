// Query latency of each optimization pass at gallery sizes 1K, 10K, and 50K, on one
// thread, one query at a time, with heap allocations counted inside the timed loop.
// Three rounds, each running every size and pass once, so passes are interleaved in
// time rather than measured back to back.
//
// Usage: bench GALLERY.bin QUERIES.bin > bench.json
#include "alloc_counter.h"
#include "scorer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace pawmatch;

constexpr size_t K = 100, WARMUP = 1000, ROUNDS = 3;
constexpr size_t SIZES[] = {1000, 10000, 50000};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: bench GALLERY.bin QUERIES.bin\n");
        return 2;
    }
    const Matrix all = load(argv[1]), queries = load(argv[2]);
    std::vector<double> ns(queries.rows);
    std::vector<Hit> out(K);
    float sink = 0;
    bool first = true;
    std::printf("{\"compiler\": \"%s\", \"k\": %zu, \"warmup_queries\": %zu, \"timed_queries\": %u, \"rounds\": %zu, \"results\": [",
                __VERSION__, K, WARMUP, queries.rows, ROUNDS);
    for (size_t round = 0; round < ROUNDS; ++round) {
        for (size_t size : SIZES) {
            const Matrix g = head(all, size);
            Scorer s(g, K);
            auto bench = [&](const char* name, auto&& query) {
                for (size_t i = 0; i < WARMUP; ++i) query(queries.row(i % queries.rows));
                const size_t allocations = g_allocations.load();
                for (size_t i = 0; i < queries.rows; ++i) {
                    const auto start = std::chrono::steady_clock::now();
                    query(queries.row(i));
                    ns[i] = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
                }
                const double per_query = double(g_allocations.load() - allocations) / queries.rows;
                const double mean = std::accumulate(ns.begin(), ns.end(), 0.0) / ns.size() / 1000;
                std::sort(ns.begin(), ns.end());
                auto us = [&](double p) { return ns[size_t(std::ceil(p * ns.size())) - 1] / 1000; };  // nearest rank
                std::printf("%s\n  {\"round\": %zu, \"version\": \"%s\", \"gallery\": %zu, \"p50_us\": %.3f, \"p99_us\": %.3f, "
                            "\"p999_us\": %.3f, \"mean_us\": %.3f, \"heap_allocations_per_query\": %.3f}",
                            first ? "" : ",", round, name, size, us(0.5), us(0.99), us(0.999), mean, per_query);
                std::fflush(stdout);
                first = false;
            };
            bench("v0_naive", [&](const float* q) { sink += topk_v0_naive(g, q, K)[0].score; });
            bench("v1_simd", [&](const float* q) { sink += topk_v1_simd(g, q, K)[0].score; });
            bench("v2_prealloc", [&](const float* q) { s.topk_v2_prealloc(q, out); sink += out[0].score; });
            bench("v3_heap", [&](const float* q) { s.topk_v3_heap(q, out); sink += out[0].score; });
            bench("v4_blocked", [&](const float* q) { s.topk_v4_blocked(q, out); sink += out[0].score; });
            bench("v5_ilp", [&](const float* q) { s.topk_v5_ilp(q, out); sink += out[0].score; });
        }
    }
    std::printf("\n]}\n");
    std::fprintf(stderr, "checksum %f\n", sink);
}
