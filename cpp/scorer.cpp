#include "scorer.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace pawmatch {

Matrix load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    uint32_t header[3];
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(header), sizeof header);
    if (!f || std::memcmp(magic, "PWMB", 4) != 0 || header[0] != 1)
        throw std::runtime_error(path + " is not a PWMB version 1 file");
    Matrix m{header[1], header[2], {}};
    m.data.resize(size_t(m.rows) * m.dim);
    f.read(reinterpret_cast<char*>(m.data.data()), std::streamsize(m.data.size() * sizeof(float)));
    if (!f) throw std::runtime_error(path + " is truncated");
    return m;
}

Matrix head(const Matrix& m, size_t rows) {
    rows = std::min<size_t>(rows, m.rows);
    return Matrix{uint32_t(rows), m.dim, std::vector<float>(m.data.begin(), m.data.begin() + rows * m.dim)};
}

float dot_scalar(const float* a, const float* b, size_t d) {
    float s = 0;
    for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

#if defined(__ARM_NEON)

float dot_simd(const float* a, const float* b, size_t d) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}

void dot4_simd(const float* q, const float* r, size_t d, float out[4]) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float32x4_t x = vld1q_f32(q + i);  // loaded once, used against four rows
        s0 = vfmaq_f32(s0, x, vld1q_f32(r + i));
        s1 = vfmaq_f32(s1, x, vld1q_f32(r + d + i));
        s2 = vfmaq_f32(s2, x, vld1q_f32(r + 2 * d + i));
        s3 = vfmaq_f32(s3, x, vld1q_f32(r + 3 * d + i));
    }
    out[0] = vaddvq_f32(s0), out[1] = vaddvq_f32(s1), out[2] = vaddvq_f32(s2), out[3] = vaddvq_f32(s3);
    for (; i < d; ++i)
        for (int j = 0; j < 4; ++j) out[j] += q[i] * r[j * d + i];
}

void dot4_ilp_simd(const float* q, const float* r, size_t d, float out[4]) {
    float32x4_t a[4][4];
    for (auto& row : a)
        for (auto& v : row) v = vdupq_n_f32(0);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const float32x4_t x[4] = {vld1q_f32(q + i), vld1q_f32(q + i + 4), vld1q_f32(q + i + 8), vld1q_f32(q + i + 12)};
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 4; ++c) a[j][c] = vfmaq_f32(a[j][c], x[c], vld1q_f32(r + j * d + i + 4 * c));
    }
    for (int j = 0; j < 4; ++j) out[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a[j][0], a[j][1]), vaddq_f32(a[j][2], a[j][3])));
    for (; i < d; ++i)
        for (int j = 0; j < 4; ++j) out[j] += q[i] * r[j * d + i];
}

#elif defined(__AVX2__) && defined(__FMA__)

static float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

float dot_simd(const float* a, const float* b, size_t d) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 32 <= d; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    float s = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}

void dot4_simd(const float* q, const float* r, size_t d, float out[4]) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 x = _mm256_loadu_ps(q + i);
        s0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(r + i), s0);
        s1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(r + d + i), s1);
        s2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(r + 2 * d + i), s2);
        s3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(r + 3 * d + i), s3);
    }
    out[0] = hsum(s0), out[1] = hsum(s1), out[2] = hsum(s2), out[3] = hsum(s3);
    for (; i < d; ++i)
        for (int j = 0; j < 4; ++j) out[j] += q[i] * r[j * d + i];
}

void dot4_ilp_simd(const float* q, const float* r, size_t d, float out[4]) {
    __m256 a[4][2];
    for (auto& row : a) row[0] = row[1] = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m256 x0 = _mm256_loadu_ps(q + i), x1 = _mm256_loadu_ps(q + i + 8);
        for (int j = 0; j < 4; ++j) {
            a[j][0] = _mm256_fmadd_ps(x0, _mm256_loadu_ps(r + j * d + i), a[j][0]);
            a[j][1] = _mm256_fmadd_ps(x1, _mm256_loadu_ps(r + j * d + i + 8), a[j][1]);
        }
    }
    for (int j = 0; j < 4; ++j) out[j] = hsum(_mm256_add_ps(a[j][0], a[j][1]));
    for (; i < d; ++i)
        for (int j = 0; j < 4; ++j) out[j] += q[i] * r[j * d + i];
}

#else

float dot_simd(const float* a, const float* b, size_t d) { return dot_scalar(a, b, d); }

void dot4_simd(const float* q, const float* r, size_t d, float out[4]) {
    for (int j = 0; j < 4; ++j) out[j] = dot_scalar(q, r + j * d, d);
}

void dot4_ilp_simd(const float* q, const float* r, size_t d, float out[4]) { dot4_simd(q, r, d, out); }

#endif

template <float (*Dot)(const float*, const float*, size_t)>
static std::vector<Hit> topk_allocating(const Matrix& g, const float* q, size_t k) {
    k = std::min<size_t>(k, g.rows);
    std::vector<float> scores(g.rows);
    for (size_t i = 0; i < g.rows; ++i) scores[i] = Dot(q, g.row(i), g.dim);
    std::vector<uint32_t> order(g.rows);
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](uint32_t a, uint32_t b) { return better({scores[a], a}, {scores[b], b}); });
    std::vector<Hit> out(k);
    for (size_t j = 0; j < k; ++j) out[j] = {scores[order[j]], order[j]};
    return out;
}

std::vector<Hit> topk_v0_naive(const Matrix& g, const float* q, size_t k) {
    return topk_allocating<dot_scalar>(g, q, k);
}

std::vector<Hit> topk_v1_simd(const Matrix& g, const float* q, size_t k) {
    return topk_allocating<dot_simd>(g, q, k);
}

Scorer::Scorer(const Matrix& g, size_t k) : g_(g), scores_(g.rows), order_(g.rows), heap_(std::min<size_t>(k, g.rows)) {}

void Scorer::topk_v2_prealloc(const float* q, std::span<Hit> out) {
    const size_t k = std::min(out.size(), order_.size());
    for (size_t i = 0; i < g_.rows; ++i) scores_[i] = dot_simd(q, g_.row(i), g_.dim);
    std::iota(order_.begin(), order_.end(), 0u);
    std::partial_sort(order_.begin(), order_.begin() + k, order_.end(),
                      [&](uint32_t a, uint32_t b) { return better({scores_[a], a}, {scores_[b], b}); });
    for (size_t j = 0; j < k; ++j) out[j] = {scores_[order_[j]], order_[j]};
}

// Bounded heap whose top is the worst kept hit: a new row costs one comparison
// unless it beats that hit. score_block(i, s) writes the scores of rows i.. into s
// and returns how many it wrote.
template <typename Score>
void Scorer::heap_select(Score score_block, std::span<Hit> out) {
    const size_t k = std::min(out.size(), heap_.size());
    if (k == 0) return;
    size_t n = 0;
    float s[4];
    for (uint32_t i = 0; i < g_.rows;) {
        const uint32_t m = score_block(i, s);
        for (uint32_t j = 0; j < m; ++j, ++i) {
            const Hit h{s[j], i};
            if (n < k) {
                heap_[n++] = h;
                std::push_heap(heap_.begin(), heap_.begin() + n, better);
            } else if (better(h, heap_[0])) {
                std::pop_heap(heap_.begin(), heap_.begin() + k, better);
                heap_[k - 1] = h;
                std::push_heap(heap_.begin(), heap_.begin() + k, better);
            }
        }
    }
    std::sort_heap(heap_.begin(), heap_.begin() + n, better);
    std::copy(heap_.begin(), heap_.begin() + n, out.begin());
}

void Scorer::topk_v3_heap(const float* q, std::span<Hit> out) {
    heap_select([&](uint32_t i, float* s) { s[0] = dot_simd(q, g_.row(i), g_.dim); return 1u; }, out);
}

template <void (*Dot4)(const float*, const float*, size_t, float*)>
static auto four_rows(const Matrix& g, const float* q) {
    return [&g, q](uint32_t i, float* s) {
        if (i + 4 <= g.rows) {
            Dot4(q, g.row(i), g.dim, s);
            return 4u;
        }
        s[0] = dot_simd(q, g.row(i), g.dim);
        return 1u;
    };
}

void Scorer::topk_v4_blocked(const float* q, std::span<Hit> out) { heap_select(four_rows<dot4_simd>(g_, q), out); }

void Scorer::topk_v5_ilp(const float* q, std::span<Hit> out) { heap_select(four_rows<dot4_ilp_simd>(g_, q), out); }

}  // namespace pawmatch
