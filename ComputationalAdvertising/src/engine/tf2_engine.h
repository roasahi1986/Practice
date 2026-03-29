// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_TF2_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_TF2_ENGINE_H_

#include <stdint.h>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <shared_mutex>
#include <unordered_set>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

#include "ComputationalAdvertising/src/engine/engine.h"

namespace computational_advertising {

struct TF2TensorMeta {
  std::string          operation_name;
  std::string          operation_type;
  std::string          device;
  int32_t              num_dims;
  std::vector<int64_t> shape;

  std::string to_string();
};

struct TF2ModelMeta {
  absl::flat_hash_map<std::string, TF2TensorMeta> input_metas;
  absl::flat_hash_map<std::string, TF2TensorMeta> output_metas;

  std::string to_string();
};

class TF2Engine : public Engine {
 public:
  explicit TF2Engine(const EngineConf& engine_conf) noexcept(false);
  virtual ~TF2Engine();

  TF2Engine() = delete;
  TF2Engine& operator=(const TF2Engine&) = delete;
  TF2Engine(const TF2Engine&) = delete;

  // Get brand of engine
  std::string brand() noexcept override;

  // Perform warmup using TF runtime
  void warmup(Instance *instance, Score *score) noexcept(false) override;

  // Perform inference using the TF runtime
  void infer(Instance *instance, Score *score) noexcept(false) override;

  // Perform inference with trace using the TF runtime
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
  virtual void set_gpu(tensorflow::ConfigProto *tf_session_conf) noexcept(false);

  // Create session
  void create_session() override;

  // Sub initialization
  void sub_init() override;

  void get_tf_tensor_meta_by_tf_operation_name(
    const std::string& tf_operation_name,
    absl::flat_hash_map<std::string, TF2TensorMeta> *tf_tensor_meta
  );  // NOLINT

  void instance_to_tensor(
    const Instance& instance, std::vector<std::pair<std::string, tensorflow::Tensor>> *input_tensors
  );  // NOLINT

  void score_from_tensor(
    const std::vector<tensorflow::Tensor>& output_tensors, Score *score
  );  // NOLINT

 protected:
  // Preventing from distructing during inference, should be gurranteed by caller
  // std::shared_mutex engine_mtx_;

  std::unordered_set<std::string> tags_;
  tensorflow::SessionOptions      session_opts_;
  tensorflow::RunOptions          run_opts_;
  tensorflow::SavedModelBundle    model_bundle_;

  TF2ModelMeta tf_model_meta_;
};

class TF2EngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<TF2EngineFactory> instance_;

 protected:
  TF2EngineFactory() = default;

 public:
  TF2EngineFactory(const TF2EngineFactory&) = delete;
  TF2EngineFactory& operator=(const TF2EngineFactory&) = delete;

  TF2EngineFactory(TF2EngineFactory&&) = delete;
  TF2EngineFactory& operator=(TF2EngineFactory&&) = delete;

  virtual ~TF2EngineFactory() {}

  static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new TF2Engine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_TF2_ENGINE_H_
