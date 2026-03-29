// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/tf2_gpu_engine.h"

#include <fstream>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace computational_advertising {

TF2GPUEngine::TF2GPUEngine(const EngineConf& engine_conf) noexcept(false) :
  TF2Engine(engine_conf) {
}

TF2GPUEngine::~TF2GPUEngine() {}

std::string TF2GPUEngine::brand() noexcept {
  return kBrandTFGPU;
}

void TF2GPUEngine::set_gpu(tensorflow::ConfigProto *tf_session_conf) noexcept(false) {
  if (nullptr == tf_session_conf) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
      + "Session config is nullptr";
    throw std::runtime_error(err_msg);
  }

  (*tf_session_conf->mutable_device_count())["GPU"] = 1;
  tensorflow::GPUOptions gpu;
  gpu.set_per_process_gpu_memory_fraction(0.0);
  gpu.set_allow_growth(true);
  // gpu.set_allocator_type("BFC");
  gpu.set_force_gpu_compatible(true);
  gpu.set_visible_device_list("0");
  tf_session_conf->mutable_gpu_options()->CopyFrom(gpu);
  tf_session_conf->set_allow_soft_placement(false);

  tags_.insert(tensorflow::kSavedModelTagGpu);
}

std::unique_ptr<TF2GPUEngineFactory> TF2GPUEngineFactory::instance_ = nullptr;
EngineFactory *TF2GPUEngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new TF2GPUEngineFactory());
  }
  return instance_.get();
}

}  // namespace computational_advertising
