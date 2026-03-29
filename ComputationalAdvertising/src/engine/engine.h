// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_ENGINE_H_

#include <stdint.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "absl/log/log.h"
#include "absl/cleanup/cleanup.h"
#include "absl/random/random.h"
#include "absl/container/flat_hash_map.h"
#include "BShoshany/BS_thread_pool.hpp"
#include "ComputationalAdvertising/src/util/functional/timer.h"
#include "ComputationalAdvertising/src/util/os/resource_used.h"
#include "ComputationalAdvertising/src/engine/sample.h"
#include "src/proto/perf.pb.h"

namespace computational_advertising {

const char kBrandTF[]       = "TensorFlow";
const char kBrandTFGPU[]    = "TensorFlow-GPU";
const char kBrandTF2[]      = "TensorFlow2";
const char kBrandTF2GPU[]   = "TensorFlow2-GPU";
const char kBrandONNX[]     = "ONNX";
const char kBrandONNXDNNL[] = "ONNX-DNNL";
const char kBrandTVM[]      = "TVM";

struct EngineConf {
  std::string name                      = "";
  std::string version                   = "";
  std::string graph_file_loc            = "";
  std::vector<std::string> input_nodes  = {};
  std::vector<std::string> output_nodes = {};

  int32_t opt_level                     = 0;
  int32_t jit_level                     = 0;

  int32_t inter_op_parallelism_threads  = 32;
  int32_t intra_op_parallelism_threads  = 1;

  bool use_global_thread_pool           = true;
  bool ort_parrallel_execution          = false;

  std::string detail() noexcept {
    return "name: " + name + ", version: " + version + ", graph_file_loc: " + graph_file_loc
      + ", input_nodes: " + std::to_string(input_nodes.size())
      + ", output_nodes: " + std::to_string(output_nodes.size())
      + ", opt_level: " + std::to_string(opt_level) + ", jit_level: " + std::to_string(jit_level)
      + ", inter_op_parallelism_threads: " + std::to_string(inter_op_parallelism_threads)
      + ", intra_op_parallelism_threads: " + std::to_string(intra_op_parallelism_threads)
      + ", use_global_thread_pool: " + std::to_string(use_global_thread_pool)
      + ", ort_parrallel_execution: " + std::to_string(ort_parrallel_execution);
  }

  std::string brief() noexcept {
    return name + ":" + version;
  }
};

class Engine {
 public:
  explicit Engine(const EngineConf& engine_conf) noexcept(false) : conf_(engine_conf), inited_(false) {}
  virtual ~Engine() {}

  Engine() = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(const Engine&) = delete;

  // Get brand of engine
  virtual std::string brand() noexcept = 0;

  // Warmup
  virtual void warmup(Instance *instance, Score *score) noexcept(false) {
    infer(instance, score);
  }

  // Perform inference
  virtual void infer(Instance *instance, Score *score) noexcept(false) = 0;

  // Perform inference with trace
  virtual void trace(Instance *instance, Score *score) noexcept(false) = 0;

  // Get input name and shape
  virtual void get_input_name_and_shape(
    absl::flat_hash_map<std::string, std::vector<int64_t>> *input_shapes
  ) noexcept(false) = 0;  // NOLINT

  // Get output name and shape
  virtual void get_output_name_and_shape(
    absl::flat_hash_map<std::string, std::vector<int64_t>> *output_shapes
  ) noexcept(false) = 0;  // NOLINT

  void perf(
    int32_t concurrency, int32_t sample_count, int32_t batch_size, PerfIndex *perf_index, bool fill_input = false
  ) noexcept(false) {  // NOLINT
    auto infer_with_timer = [](Engine *engine, Sample *sample, double *cost_ms) {
      Timer timer;
      auto timer_cleanup = absl::MakeCleanup([&]() {
        *cost_ms = timer.f64_elapsed_ms();
      });
      engine->infer(&(sample->instance), &(sample->score));
    };

    std::vector<Sample> samples;
    random_sample_gen(&samples, sample_count, batch_size, fill_input);

    // warmup
    if (samples.size() > 0) {
      Timer timer;
      auto timer_cleanup = absl::MakeCleanup([&]() {
        LOG(INFO) << "warmup cost: " << timer.f64_elapsed_ms() << " ms";
      });
      this->warmup(&(samples[0].instance), &(samples[0].score));
    }

    // trace
    std::vector<double> cost_ms(samples.size());
    BS::thread_pool works(concurrency);
    for (int32_t i = 0; i < static_cast<int32_t>(samples.size()); ++i) {
      works.push_task(infer_with_timer, this, &(samples.at(i)), &(cost_ms[i]));
      if (static_cast<int32_t>(samples.size()) >> 1 == i) {
        this->trace(&(samples[0].instance), &(samples[0].score));
      }
    }
    works.wait_for_tasks();

    struct ResourceUsed resource_base, resource_curr;
    get_process_resource_used(&resource_base);
    Timer timer;
    for (int32_t i = 0; i < samples.size(); ++i) {
      works.push_task(infer_with_timer, this, &(samples.at(i)), &(cost_ms[i]));
    }
    works.wait_for_tasks();
    double total_cost_sec = timer.f64_elapsed_sec();
    get_process_resource_used(&resource_curr);

    std::sort(cost_ms.begin(), cost_ms.end());
    perf_index->set_cpu_usage(
      ((resource_curr.user_time - resource_base.user_time) +
      (resource_curr.system_time - resource_base.system_time)) / total_cost_sec
    );  // NOLINT
    perf_index->set_mem_usage(resource_curr.resident_mb);
    perf_index->set_cost_avg_ms(std::accumulate(cost_ms.begin(), cost_ms.end(), 0.0) / cost_ms.size());
    perf_index->set_cost_p99_ms(cost_ms[static_cast<int32_t>(cost_ms.size() * 0.99)]);
    perf_index->set_throughput(static_cast<double>(samples.size()) / total_cost_sec * batch_size);
  }

  void random_sample_gen(
    std::vector<Sample> *samples, int32_t sample_count, int32_t batch_size, bool fill_input = false
  ) noexcept(false) {  // NOLINT
    absl::flat_hash_map<std::string, std::vector<int64_t>> input_shapes;
    get_input_name_and_shape(&input_shapes);
    absl::flat_hash_map<std::string, std::vector<int64_t>> output_shapes;
    get_output_name_and_shape(&output_shapes);

    samples->resize(sample_count);
    std::random_device rd;
    absl::BitGen bitgen;
    BS::thread_pool works(16);
    for (auto& sample : *samples) {
      works.push_task([&]() {
        sample.instance.features.resize(input_shapes.size());
        int32_t i = 0;
        for (const auto& input : input_shapes) {
          auto& feature = sample.instance.features[i++];
          feature.batch_size = batch_size;
          int64_t data_size = batch_size;
          for (int32_t i = 1; i < static_cast<int32_t>(input.second.size()); ++i) {
            data_size *= input.second[i];
          }
          feature.name = input.first;
          feature.data.resize(data_size);

          // fill random data
          if (fill_input) {
            for (auto& data : feature.data) {
              data = absl::Uniform(absl::IntervalClosedClosed, bitgen, 0.0f, 1.0f);;
            }
          }
        }

        sample.score.targets.resize(output_shapes.size());
        int32_t j = 0;
        for (const auto& output : output_shapes) {
          auto& target = sample.score.targets[j++];
          target.name = output.first;
          target.batch_size = batch_size;
        }
      });
    }
    works.wait_for_tasks();
  }

  // Initialize engine
  void init() {
    load();
    build();
    set_session_options();
    create_session();
    sub_init();
    inited_ = true;
  }

 protected:
  // Load the TensorFlow graph from the .pb file
  virtual void load() = 0;

  // Build engine
  virtual void build() = 0;

  // Set session options
  virtual void set_session_options() = 0;

  // Create session
  virtual void create_session() = 0;

  // Personalized initialization for subclasses
  virtual void sub_init() {}

  EngineConf conf_;
  bool inited_;
};

class EngineFactory {
 private:
  // static std::unique_ptr<EngineFactory> instance_;

 protected:
  EngineFactory() = default;

 public:
  EngineFactory(const EngineFactory&) = delete;
  EngineFactory& operator=(const EngineFactory&) = delete;

  EngineFactory(EngineFactory&&) = delete;
  EngineFactory& operator=(EngineFactory&&) = delete;

  virtual ~EngineFactory() {}

  // static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) = 0;
};

// std::unique_ptr<EngineFactory> EngineFactory::instance_ = nullptr;
// EngineFactory *EngineFactory::instance() {
//   if (nullptr == instance_) {
//     instance_.reset(new EngineFactory());
//   }
//   return instance_.get();
// }

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_ENGINE_H_
