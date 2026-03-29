// Copyright (C) 2021 lusyu1986@icloud.com

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "benchmark/benchmark.h"
#include "ComputationalAdvertising/src/engine/sample.h"
#include "ComputationalAdvertising/src/engine/onnx_engine.h"
#include "ComputationalAdvertising/src/population/model_spec.h"
#include "ComputationalAdvertising/src/population/sample_gen.h"

const int32_t kTestDataSize = 20;
const int32_t kBatchSize = 128;
static std::vector<computational_advertising::Sample> g_samples;

static void do_setup(const benchmark::State& state) {
  FLAGS_logbufsecs = 0;
  FLAGS_minloglevel = google::ERROR;
  FLAGS_logtostdout = true;

  std::string meta_file = "data/models/model_2/model_conf.json";
  computational_advertising::ModelMeta model_meta;
  model_meta.load(meta_file);
  computational_advertising::random_sample_gen(model_meta, &g_samples, kTestDataSize, kBatchSize);
}

static void do_teardown(const benchmark::State& state) {
}

static void bm_onnx_engine(benchmark::State& state) {  // NOLINT
  computational_advertising::EngineConf engine_conf {
    .name = "model_2",
    .version = "1.0.0",
    .graph_file_loc = "data/models/model_2/1/graph.onnx",
    .input_nodes = {"dense", "onehot", "sparse_input_folded", "sparse_input_unfolded"},
    .output_nodes = {"predict_node", "p0_click", "p0_atc", "p0_order"},
    .opt_level = static_cast<int32_t>(state.range(0)),
    .jit_level = static_cast<int32_t>(state.range(1)),
    .inter_op_parallelism_threads = static_cast<int32_t>(state.range(2)),
    .intra_op_parallelism_threads = static_cast<int32_t>(state.range(3))
  };
  std::unique_ptr<computational_advertising::ONNXEngine> engine(new computational_advertising::ONNXEngine(engine_conf));

  for (auto _ : state) {
    for (auto& sample : g_samples) {
      engine->infer(&sample.instance, &sample.score);
      benchmark::ClobberMemory();
    }
  }
}

BENCHMARK(bm_onnx_engine)
  ->Args({0, 0, 1, 1})
  ->Args({0, 0, 1, 8})
  ->Args({0, 0, 8, 1})
  ->Args({0, 0, 8, 8})
  ->Args({0, 1, 1, 1})
  ->Args({0, 1, 1, 8})
  ->Args({0, 1, 8, 1})
  ->Args({0, 1, 8, 8})
  ->Args({0, 2, 1, 1})
  ->Args({0, 2, 1, 8})
  ->Args({0, 2, 8, 1})
  ->Args({0, 2, 8, 8})
  ->Args({1, 0, 1, 1})
  ->Args({1, 0, 1, 8})
  ->Args({1, 0, 8, 1})
  ->Args({1, 0, 8, 8})
  ->Args({1, 1, 1, 1})
  ->Args({1, 1, 1, 8})
  ->Args({1, 1, 8, 1})
  ->Args({1, 1, 8, 8})
  ->Args({1, 2, 1, 1})
  ->Args({1, 2, 1, 8})
  ->Args({1, 2, 8, 1})
  ->Args({1, 2, 8, 8})
  ->Setup(do_setup)
  ->Teardown(do_teardown)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK_MAIN();
