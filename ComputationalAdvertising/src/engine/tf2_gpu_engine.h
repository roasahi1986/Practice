// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_TF2_GPU_ENGINE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_TF2_GPU_ENGINE_H_

#include <memory>
#include <string>
#include "ComputationalAdvertising/src/engine/tf2_engine.h"

namespace computational_advertising {

class TF2GPUEngine : public TF2Engine {
 public:
  explicit TF2GPUEngine(const EngineConf& engine_conf) noexcept(false);
  virtual ~TF2GPUEngine();

  TF2GPUEngine() = delete;
  TF2GPUEngine& operator=(const TF2GPUEngine&) = delete;
  TF2GPUEngine(const TF2GPUEngine&) = delete;

  // Get brand of engine
  std::string brand() noexcept override;

 protected:
  // Set session options
  void set_gpu(tensorflow::ConfigProto *tf_session_conf) noexcept(false) override;
};

class TF2GPUEngineFactory : public EngineFactory {
 private:
  static std::unique_ptr<TF2GPUEngineFactory> instance_;

 protected:
  TF2GPUEngineFactory() = default;

 public:
  TF2GPUEngineFactory(const TF2GPUEngineFactory&) = delete;
  TF2GPUEngineFactory& operator=(const TF2GPUEngineFactory&) = delete;

  TF2GPUEngineFactory(TF2GPUEngineFactory&&) = delete;
  TF2GPUEngineFactory& operator=(TF2GPUEngineFactory&&) = delete;

  virtual ~TF2GPUEngineFactory() {}

  static EngineFactory *instance();

  virtual Engine *create(const EngineConf& engine_conf) noexcept(false) {
    Engine *engine = new TF2GPUEngine(engine_conf);
    engine->init();
    return engine;
  }
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_TF2_GPU_ENGINE_H_
