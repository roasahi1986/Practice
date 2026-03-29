// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/onnx_dnnl_engine.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "onnxruntime/dnnl_provider_options.h"

namespace computational_advertising {

ONNXDNNLEngine::ONNXDNNLEngine(const EngineConf& engine_conf) noexcept(false) :
  ONNXEngine(engine_conf) {
}

ONNXDNNLEngine::~ONNXDNNLEngine() {}

std::string ONNXDNNLEngine::brand() noexcept {
  return kBrandONNXDNNL;
}

void ONNXDNNLEngine::set_session_options() {
  ONNXEngine::set_session_options();

  OrtDnnlProviderOptions dnnl_options = {
    .use_arena = true,
    .threadpool_args = nullptr
  };
  session_opts_->AppendExecutionProvider_Dnnl(dnnl_options);
  LOG(INFO) << "Set DNNL as execution provider";
}

std::unique_ptr<ONNXDNNLEngineFactory> ONNXDNNLEngineFactory::instance_ = nullptr;
EngineFactory *ONNXDNNLEngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new ONNXDNNLEngineFactory();
  }
  return instance_.get();
}

}  // namespace computational_advertising
