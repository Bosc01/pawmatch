// Counts global heap allocations. Include in exactly one translation unit per binary.
#pragma once

#include <atomic>
#include <cstdlib>
#include <new>

inline std::atomic<size_t> g_allocations{0};

void* operator new(std::size_t n) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
