// Copyright (C) 2021 lusyu1986@icloud.com

#include <algorithm>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "benchmark/benchmark.h"

static std::default_random_engine g_rng;
static std::uniform_int_distribution<uint64_t> g_dis;

static void do_setup(const benchmark::State& state) {
  g_rng = std::default_random_engine {};
  g_dis = std::uniform_int_distribution<uint64_t>(
    std::numeric_limits<std::uint64_t>::min(),
    std::numeric_limits<std::uint64_t>::max()
  );  // NOLINT
}

static void do_teardown(const benchmark::State& state) {
}

static void bm_unordered_map_find(benchmark::State& state) {  // NOLINT
  const int32_t num_keys = state.range(0);
  std::unordered_map<uint64_t, uint64_t> um_ins;
  std::vector<uint64_t> keys;

  for (int32_t i = 0; i < num_keys; ++i) {
    keys.push_back(g_dis(g_rng));
    um_ins.insert(std::make_pair(keys.back(), static_cast<uint64_t>(i)));
  }

  std::shuffle(keys.begin(), keys.end(), g_rng);

  for (auto _ : state) {
    for (auto& key : keys) {
      benchmark::DoNotOptimize(um_ins.find(key));
      benchmark::ClobberMemory();
    }
  }
}

static void bm_flat_hash_map_find(benchmark::State& state) {  // NOLINT
  const int32_t num_keys = state.range(0);

  absl::flat_hash_map<uint64_t, uint64_t> fhm_ins;
  std::vector<uint64_t> keys;

  for (int32_t i = 0; i < num_keys; ++i) {
    keys.push_back(g_dis(g_rng));
    fhm_ins.insert(std::make_pair(keys.back(), static_cast<uint64_t>(i)));
  }

  std::shuffle(keys.begin(), keys.end(), g_rng);

  for (auto _ : state) {
    for (auto& key : keys) {
      benchmark::DoNotOptimize(fhm_ins.find(key));
      benchmark::ClobberMemory();
    }
  }
}

BENCHMARK(bm_unordered_map_find)
  ->Args({10000ll})
  ->Args({1000000ll})
  ->Setup(do_setup)
  ->Teardown(do_teardown)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(bm_flat_hash_map_find)
  ->Args({10000ll})
  ->Args({1000000ll})
  ->Setup(do_setup)
  ->Teardown(do_teardown)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK_MAIN();
