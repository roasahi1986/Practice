// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_TF_GPU_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_TF_GPU_ENGINE_H_

#include <memory>
#include <string>
#include "tensorflow/c/c_api.h"
#include "ComputationalAdvertising/src/engine/tf_engine.h"

namespace computational_advertising {

class TFGPUEngine : public TFEngine {
 public:
  explicit TFGPUEngine(const EngineConf& engine_conf) noexcept(false);
  virtual ~TFGPUEngine();

  TFGPUEngine() = delete;
  TFGPUEngine& operator=(const TFGPUEngine&) = delete;
  TFGPUEngine(const TFGPUEngine&) = delete;

  // Get brand of engine
  std::string brand() noexcept override;

 protected:
  // Set session options
  void set_gpu(tensorflow::ConfigProto *tf_session_conf) noexcept(false) override;
};

class TFGPUEngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<TFGPUEngineFactory> instance_;

 protected:
  TFGPUEngineFactory() = default;

 public:
  TFGPUEngineFactory(const TFGPUEngineFactory&) = delete;
  TFGPUEngineFactory& operator=(const TFGPUEngineFactory&) = delete;

  TFGPUEngineFactory(TFGPUEngineFactory&&) = delete;
  TFGPUEngineFactory& operator=(TFGPUEngineFactory&&) = delete;

  virtual ~TFGPUEngineFactory() {}

  static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new TFGPUEngine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_TF_GPU_ENGINE_H_
