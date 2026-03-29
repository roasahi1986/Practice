// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_TVM_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_TVM_ENGINE_H_

#include <memory>
#include <vector>
#include <string>
#include <shared_mutex>
#include "absl/container/flat_hash_map.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"
#include "ComputationalAdvertising/src/engine/engine.h"

namespace computational_advertising {

class TVMEngine : public Engine {
 public:
  explicit TVMEngine(const EngineConf& engine_conf) noexcept(false);
  virtual ~TVMEngine();

  TVMEngine() = delete;
  TVMEngine& operator=(const TVMEngine&) = delete;
  TVMEngine(const TVMEngine&) = delete;

  // Get brand of engine
  std::string brand() noexcept override;

  // Perform inference using the ONNX runtime
  void infer(Instance *instance, Score *score) noexcept(false) override;

  // Perform inference with trace using the ONNX runtime
  void trace(Instance *instance, Score *score) noexcept(false) override;

  // Get input name and shape
  void get_input_name_and_shape(
    absl::flat_hash_map<std::string, std::vector<int64_t>> *input_shapes
  ) noexcept(false) override;  // NOLINT

  // Get output name and shape
  void get_output_name_and_shape(
    absl::flat_hash_map<std::string, std::vector<int64_t>> *output_shapes
  ) noexcept(false) override;  // NOLINT

 protected:
  // Load the TensorFlow graph from the .pb file
  void load() override;

  // Build engine
  void build() override;

  // Set session options
  void set_session_options() override;

  // Create session
  void create_session() override;

  // Sub initialization
  void sub_init() override;

 protected:
  // Preventing from distructing during inference, should be gurranteed by caller
  // std::shared_mutex engine_mtx_;

  int32_t dtype_code_;
  int32_t dtype_bits_;
  int32_t dtype_lanes_;
  int32_t device_type_;
  int32_t device_id_;
  tvm::runtime::Module module_;
  tvm::runtime::PackedFunc set_input_;
  tvm::runtime::PackedFunc run_;
  tvm::runtime::PackedFunc get_output_;
  tvm::runtime::PackedFunc release_;

  absl::flat_hash_map<std::string, std::vector<int64_t>> input_shapes_;
  absl::flat_hash_map<std::string, std::vector<int64_t>> output_shapes_;
};

class TVMEngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<TVMEngineFactory> instance_;

 protected:
  TVMEngineFactory() = default;

 public:
  TVMEngineFactory(const TVMEngineFactory&) = delete;
  TVMEngineFactory& operator=(const TVMEngineFactory&) = delete;

  TVMEngineFactory(TVMEngineFactory&&) = delete;
  TVMEngineFactory& operator=(TVMEngineFactory&&) = delete;

  virtual ~TVMEngineFactory() {}

  static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new TVMEngine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_TVM_ENGINE_H_
