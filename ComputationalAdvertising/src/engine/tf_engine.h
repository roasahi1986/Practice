// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_TF_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_TF_ENGINE_H_

#include <stdint.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <shared_mutex>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/c/c_api.h"
#include "ComputationalAdvertising/src/engine/engine.h"

namespace computational_advertising {

const char kGlobalInterOpThreadPool[] = "GlobalTensorFlowInterOpThreadPool";

struct TFTensorMeta {
  TF_Output           *output = nullptr;
  std::string          operation_name;
  std::string          operation_type;
  int32_t              operation_num_inputs;
  int32_t              operation_num_outputs;
  TF_DataType          data_type;
  int32_t              data_size;
  int32_t              num_dims;
  std::vector<int64_t> shape;
  size_t               instance_size;
  int32_t              index;

  std::string to_string();
};

struct TFModelMeta {
  absl::flat_hash_map<std::string, TFTensorMeta> input_metas;
  absl::flat_hash_map<std::string, TFTensorMeta> output_metas;
  std::vector<TF_Output> input_specs;
  std::vector<TF_Output> output_specs;

  std::string to_string();
};

class TFEngine : public Engine {
 public:
  explicit TFEngine(const EngineConf& engine_conf) noexcept(false);
  virtual ~TFEngine();

  TFEngine() = delete;
  TFEngine& operator=(const TFEngine&) = delete;
  TFEngine(const TFEngine&) = delete;

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

  // Run session
  void run_session(
    std::vector<TF_Tensor*> *input_tensors, std::vector<TF_Tensor*> *output_tensors,
    TF_Buffer *tf_run_opts = nullptr, TF_Buffer *tf_metadata = nullptr
  );  // NOLINT

  // Iterate through the operations in the graph
  void iterate_through_operations(std::function<void(TF_Operation*)> do_something_with_operation);

  // Get TFTensorMeta by TF_Operation name
  void get_tf_tensor_meta_by_tf_operation_name(
    const std::string& tf_operation_name,
    absl::flat_hash_map<std::string, TFTensorMeta> *tf_tensor_meta,
    int32_t *index = nullptr
  );  // NOLINT

  // Convert TF_Output to TFTensorMeta
  void convert_tf_output_to_tf_tensor_meta(
    const TF_Output& tf_output, TFTensorMeta *tf_tensor_meta
  );  // NOLINT

  // Get input and output ops
  // void get_input_output_ops();

  // Print graph information
  // void print_graph_info();

  void instance_to_tensor(
    Instance *instance, std::vector<TF_Tensor*> *input_tensors
  ) noexcept(false);  // NOLINT
  void score_from_tensor(
    const std::vector<TF_Tensor*>& output_tensors, Score *score
  ) noexcept(false);  // NOLINT

 protected:
  // Preventing from distructing during inference, should be gurranteed by caller
  // std::shared_mutex engine_mtx_;

  TFModelMeta tf_model_meta_;

  TF_Buffer         *graph_buffer_;
  TF_Graph          *graph_;
  TF_SessionOptions *session_opts_;
  TF_Session        *session_;
  TF_Buffer         *default_run_option_buf_;
  TF_Buffer         *warmup_run_option_buf_;
};

class TFEngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<TFEngineFactory> instance_;

 protected:
  TFEngineFactory() = default;

 public:
  TFEngineFactory(const TFEngineFactory&) = delete;
  TFEngineFactory& operator=(const TFEngineFactory&) = delete;

  TFEngineFactory(TFEngineFactory&&) = delete;
  TFEngineFactory& operator=(TFEngineFactory&&) = delete;

  virtual ~TFEngineFactory() {}

  static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new TFEngine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_TF_ENGINE_H_
