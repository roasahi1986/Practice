// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_ONNX_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_ONNX_ENGINE_H_

#include <memory>
#include <vector>
#include <string>
#include <shared_mutex>
#include "absl/container/flat_hash_map.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "ComputationalAdvertising/src/engine/engine.h"

namespace computational_advertising {

struct ONNXTensorMeta {
  std::string          name;
  int32_t              num_dims;
  std::vector<int64_t> shape;
  size_t               instance_size;
  int32_t              index;

  std::string to_string();
};

struct ONNXModelMeta {
  absl::flat_hash_map<std::string, ONNXTensorMeta> input_metas;
  absl::flat_hash_map<std::string, ONNXTensorMeta> output_metas;

  std::string to_string();
};

class ONNXEngine : public Engine {
 public:
  explicit ONNXEngine(const EngineConf& engine_conf) noexcept(false);
  virtual ~ONNXEngine();

  ONNXEngine() = delete;
  ONNXEngine& operator=(const ONNXEngine&) = delete;
  ONNXEngine(const ONNXEngine&) = delete;

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

  void run_session(Instance *instance, Score *score, Ort::Session *session) noexcept(false);

 protected:
  // Preventing from distructing during inference, should be gurranteed by caller
  // std::shared_mutex engine_mtx_;

  Ort::Env            *env_;
  Ort::Session        *session_;
  Ort::SessionOptions *session_opts_;

  ONNXModelMeta onnx_model_meta_;
};

class ONNXEngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<ONNXEngineFactory> instance_;

 protected:
  ONNXEngineFactory() = default;

 public:
  ONNXEngineFactory(const ONNXEngineFactory&) = delete;
  ONNXEngineFactory& operator=(const ONNXEngineFactory&) = delete;

  ONNXEngineFactory(ONNXEngineFactory&&) = delete;
  ONNXEngineFactory& operator=(ONNXEngineFactory&&) = delete;

  virtual ~ONNXEngineFactory() {}

  static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new ONNXEngine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_ONNX_ENGINE_H_
