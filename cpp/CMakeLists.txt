cmake_minimum_required(VERSION 3.20)
project(pawmatch_scorer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
# NEON is baseline on arm64. On x86-64, build the AVX2 and FMA path.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>")
endif()

add_library(scorer scorer.cpp)
add_executable(bench bench.cpp)
target_link_libraries(bench scorer)

# The CUDA benchmark builds only where a CUDA compiler is found.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  add_executable(bench_cuda bench_cuda.cu)
  target_link_libraries(bench_cuda scorer)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(scorer_test test_scorer.cpp)
target_link_libraries(scorer_test scorer GTest::gtest_main)
target_compile_definitions(scorer_test PRIVATE PAWMATCH_M4_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../data/cache/m4")
include(GoogleTest)
gtest_discover_tests(scorer_test DISCOVERY_TIMEOUT 60)
