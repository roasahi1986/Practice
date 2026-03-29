// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_TRT_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_TRT_ENGINE_H_

#include <memory>
#include <vector>
#include <string>
#include <shared_mutex>
#include "absl/container/flat_hash_map.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "ComputationalAdvertising/src/engine/engine.h"

namespace computational_advertising {

class TRTEngine : public Engine {
 public:
  explicit TRTEngine(const EngineConf& engine_conf) noexcept(false);
  virtual ~TRTEngine();

  TRTEngine() = delete;
  TRTEngine& operator=(const TRTEngine&) = delete;
  TRTEngine(const TRTEngine&) = delete;

  // Get brand of engine
  std::string brand() noexcept override;

  // Perform inference using the TRT runtime
  void infer(Instance *instance, Score *score) noexcept(false) override;

  // Perform inference with trace using the TRT runtime
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

  nvinfer1::Logger              logger_;
  nvinfer1::IBuilder           *builder_;
  nvinfer1::INetworkDefinition *network_;
  nvinfer1::IBuilderConfig     *config_;
  nvinfer1::IOnnxParser        *parser_;
  nvinfer1::ICudaEngine        *engine_;
  nvinfer1::IExecutionContext  *context_;
};

class TRTEngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<TRTEngineFactory>   instance_;
  static std::unique_ptr<nvinfer1::IRuntime> runtime_;

 protected:
  TRTEngineFactory() = default;

 public:
  TRTEngineFactory(const TRTEngineFactory&) = delete;
  TRTEngineFactory& operator=(const TRTEngineFactory&) = delete;

  TRTEngineFactory(TRTEngineFactory&&) = delete;
  TRTEngineFactory& operator=(TRTEngineFactory&&) = delete;

  virtual ~TRTEngineFactory() {}

  static EngineFactory *instance();
  nvinfer1::IRuntime *runtime() { return runtime_.get(); }

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new TRTEngine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_TRT_ENGINE_H_
