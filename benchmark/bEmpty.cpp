#include <benchmark/benchmark.h>

static void BM_Empty(benchmark::State &state) {
  for (auto _ : state) {
    // Empty benchmark
  }
}
BENCHMARK(BM_Empty);
