// Parity with the Python reference, agreement between passes, and a hot path with
// no heap allocation. Fixtures come from `python -m pawmatch.export`.
#include "alloc_counter.h"
#include "scorer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

using namespace pawmatch;

namespace {

constexpr size_t K = 100;

std::string fixture(const char* name) {
    const char* dir = std::getenv("PAWMATCH_M4_DIR");
    return std::string(dir ? dir : PAWMATCH_M4_DIR_DEFAULT) + "/" + name;
}

struct Data {
    Matrix gallery = load(fixture("gallery.bin"));
    Matrix queries = load(fixture("queries.bin"));
    // Python float64 scores of every query against the first reference.dim gallery rows.
    Matrix reference = load(fixture("reference_scores.bin"));
    Matrix parity = head(gallery, reference.dim);
};

const Data& data() {
    static const Data d;
    return d;
}

}  // namespace

TEST(Parity, ScoresMatchPythonWithin1e5On10000Queries) {
    const Data& d = data();
    ASSERT_EQ(d.queries.rows, 10000u);
    ASSERT_EQ(d.reference.rows, d.queries.rows);
    ASSERT_EQ(d.parity.rows % 4, 0u);
    double worst[3] = {0, 0, 0};  // scalar, simd, four-row simd
    float four[4];
    for (size_t i = 0; i < d.queries.rows; ++i) {
        const float* q = d.queries.row(i);
        const float* ref = d.reference.row(i);
        for (size_t j = 0; j < d.parity.rows; j += 4) {
            dot4_simd(q, d.parity.row(j), d.parity.dim, four);
            for (size_t r = 0; r < 4; ++r) {
                worst[0] = std::max(worst[0], double(std::abs(dot_scalar(q, d.parity.row(j + r), d.parity.dim) - ref[j + r])));
                worst[1] = std::max(worst[1], double(std::abs(dot_simd(q, d.parity.row(j + r), d.parity.dim) - ref[j + r])));
                worst[2] = std::max(worst[2], double(std::abs(four[r] - ref[j + r])));
            }
        }
    }
    std::printf("max abs error against Python: scalar %.3g, simd %.3g, four-row simd %.3g\n", worst[0], worst[1], worst[2]);
    for (double w : worst) EXPECT_LE(w, 1e-5);
}

TEST(TopK, EveryPassMatchesPythonTopKScores) {
    const Data& d = data();
    Scorer s(d.parity, K);
    std::vector<Hit> v2(K), v3(K), v4(K);
    std::vector<float> ref;
    for (size_t i = 0; i < d.queries.rows; ++i) {
        const float* q = d.queries.row(i);
        ref.assign(d.reference.row(i), d.reference.row(i) + d.reference.dim);
        std::partial_sort(ref.begin(), ref.begin() + K, ref.end(), std::greater<>());
        const auto v0 = topk_v0_naive(d.parity, q, K), v1 = topk_v1_simd(d.parity, q, K);
        s.topk_v2_prealloc(q, v2);
        s.topk_v3_heap(q, v3);
        s.topk_v4_blocked(q, v4);
        for (const std::vector<Hit>* hits : std::initializer_list<const std::vector<Hit>*>{&v0, &v1, &v2, &v3, &v4})
            for (size_t j = 0; j < K; ++j) ASSERT_NEAR((*hits)[j].score, ref[j], 1e-5) << "query " << i << " rank " << j;
    }
}

TEST(TopK, HeapPassesMatchSortOn50KGallery) {
    const Data& d = data();
    Scorer s(d.gallery, K);
    std::vector<Hit> v2(K), v3(K), v4(K);
    for (size_t i = 0; i < 200; ++i) {
        const float* q = d.queries.row(i);
        s.topk_v2_prealloc(q, v2);
        s.topk_v3_heap(q, v3);
        s.topk_v4_blocked(q, v4);
        for (size_t j = 0; j < K; ++j) {
            ASSERT_EQ(v3[j].index, v2[j].index) << "query " << i << " rank " << j;  // same dot product
            ASSERT_EQ(v3[j].score, v2[j].score);
            ASSERT_NEAR(v4[j].score, v3[j].score, 1e-6);  // different summation order
        }
    }
}

TEST(Allocations, PreallocatedPassesAllocateNothingPerQuery) {
    const Data& d = data();
    Scorer s(d.gallery, K);
    std::vector<Hit> out(K);
    size_t before = g_allocations.load();
    for (size_t i = 0; i < 200; ++i) {
        s.topk_v2_prealloc(d.queries.row(i), out);
        s.topk_v3_heap(d.queries.row(i), out);
        s.topk_v4_blocked(d.queries.row(i), out);
    }
    EXPECT_EQ(g_allocations.load() - before, 0u);
    before = g_allocations.load();  // the counter itself works: v0 allocates on every query
    (void)topk_v0_naive(d.gallery, d.queries.row(0), K);
    EXPECT_GT(g_allocations.load() - before, 0u);
}

TEST(Load, RejectsAFileWithoutTheHeader) {
    EXPECT_THROW(load(fixture("manifest.json")), std::runtime_error);
}
