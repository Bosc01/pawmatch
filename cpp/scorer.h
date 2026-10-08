// Brute-force inner-product scorer for L2-normalized embeddings (Milestone 4).
//
// The versions are successive optimization passes, kept side by side so the
// benchmark measures each one under the same conditions. All return the top k
// gallery rows by score, best first, with ties going to the lower row index.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pawmatch {

// Row-major float32 matrix, loaded from the PWMB flat binary format:
// "PWMB", uint32 version 1, uint32 rows, uint32 dim, then rows x dim float32.
struct Matrix {
    uint32_t rows = 0, dim = 0;
    std::vector<float> data;
    const float* row(size_t i) const { return data.data() + i * dim; }
};

Matrix load(const std::string& path);
Matrix head(const Matrix& m, size_t rows);  // copy of the first rows

struct Hit {
    float score;
    uint32_t index;
};

inline bool better(const Hit& a, const Hit& b) {
    return a.score > b.score || (a.score == b.score && a.index < b.index);
}

float dot_scalar(const float* a, const float* b, size_t d);
float dot_simd(const float* a, const float* b, size_t d);  // NEON or AVX2+FMA, scalar otherwise
void dot4_simd(const float* q, const float* rows, size_t d, float out[4]);  // four consecutive rows

// v0: scalar dot products, score and index vectors allocated per query, partial_sort.
std::vector<Hit> topk_v0_naive(const Matrix& g, const float* q, size_t k);
// v1: v0 with SIMD dot products.
std::vector<Hit> topk_v1_simd(const Matrix& g, const float* q, size_t k);

// v2 to v4 allocate their buffers once, in the constructor, and write into out.
class Scorer {
public:
    Scorer(const Matrix& g, size_t k);
    void topk_v2_prealloc(const float* q, std::span<Hit> out);  // v1 without per-query allocation
    void topk_v3_heap(const float* q, std::span<Hit> out);      // bounded heap instead of score buffer and sort
    void topk_v4_blocked(const float* q, std::span<Hit> out);   // v3 scoring four rows per pass

private:
    template <typename Score>
    void heap_select(Score score_block, std::span<Hit> out);

    const Matrix& g_;
    std::vector<float> scores_;
    std::vector<uint32_t> order_;
    std::vector<Hit> heap_;
};

}  // namespace pawmatch
