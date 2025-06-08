#include <benchmark/benchmark.h>

static void BM_Linear(benchmark::State &state) {
  for (auto _ : state) {
    // Linear time complexity benchmark
    for (int i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(i);
    }
  }
}
BENCHMARK(BM_Linear)->Range(1 << 10, 1 << 20);
