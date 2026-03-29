// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/trt_engine.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace computational_advertising {

TRTEngine::TRTEngine(const EngineConf& engine_conf) noexcept(false) :
  Engine(engine_conf),
  // engine_mtx_(),
  logger_(),
  builder_(nullptr),
  network_(nullptr),
  config_(nullptr),
  parser_(nullptr),
  engine_(nullptr),
  context_(nullptr),
  runtime_(nullptr) {
}

TRTEngine::~TRTEngine() {
  try {
    // std::unique_lock<std::shared_mutex> engine_lock(engine_mtx_);
    inited_ = false;
    if (nullptr != context_) {
      context_->destroy();
      context_ = nullptr;
    }
    if (nullptr != engine_) {
      engine_->destroy();
      engine_ = nullptr;
    }
    if (nullptr != network_) {
      network_->destroy();
      network_ = nullptr;
    }
    if (nullptr != parser_) {
      parser_->destroy();
      parser_ = nullptr;
    }
    if (nullptr != config_) {
      config_->destroy();
      config_ = nullptr;
    }
    if (nullptr != builder_) {
      builder_->destroy();
      builder_ = nullptr;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  } catch (...) {
    LOG(ERROR) << "Unknown exception";
  }
}

std::string TRTEngine::brand() noexcept {
  return kBrandTRT;
}

  // Perform inference using the TRT runtime
void TRTEngine::infer(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  run_session(instance, score, session_);
}

void TRTEngine::run_session(Instance *instance, Score *score, Ort::Session *session) noexcept(false) {
  // Allocate input and output buffers
  // const int inputIndex = network->getInput(0)->getIndex();
  // const int outputIndex = network->getOutput(0)->getIndex();
  // const int batchSize = 1;
  // const int inputSize = batchSize * ...; // Specify input size
  // const int outputSize = batchSize * ...; // Specify output size
  // void* inputBuffer, * outputBuffer;
  // cudaMalloc(&inputBuffer, inputSize);
  // cudaMalloc(&outputBuffer, outputSize);

  // // Run inference
  // context->execute(batchSize, { inputBuffer }, { outputBuffer });

  // // Process the output
  // float* outputData = new float[outputSize];
  // cudaMemcpy(outputData, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
  // // TODO: Process the output data

  // // Clean up
  // delete[] outputData;
  // cudaFree(inputBuffer);
  // cudaFree(outputBuffer);
}

void TRTEngine::trace(Instance *instance, Score *score) noexcept(false) {
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }
}

void TRTEngine::load() {
  // TODO(lusyu1986@icloud.com):
  //   use mutex to protect
  // TRTEngineFactory::instance()->runtime()->setDevice(0);

  // if conf_.graph_file_loc is a .onnx file
  if (conf_.graph_file_loc.find(".onnx") != std::string::npos) {
    builder_ = nvinfer1::createInferBuilder(logger_);
    builder_->setMaxBatchSize(256);
    // builder->setMaxThreads(1024);
    // builder->platformHasFastFp16() or builder->platformHasTf32()

    network_ = builder->createNetworkV2(0U);
    config_  = builder->createBuilderConfig();
    config_->setFlag(nvinfer1::BuilderFlag::kREFIT);
    config_->setMaxWorkspaceSize(1 << 30);
    // config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    // config_->clearFlag(nvinfer1::BuilderFlag::kFP16);
    // config_->setFlag(nvinfer1::BuilderFlag::kTF32);
    // config_->clearFlag(nvinfer1::BuilderFlag::kTF32);

    parser_ = nvinfer1::createParser(*network_, gLogger);
    parser->parseFromFile(conf.graph_file_loc.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    engine_ = builder_->buildEngineWithConfig(*network_, *config_);
  } else {
    std::ifstream model_in(conf_.graph_file_loc, std::ios::binary);
    auto model_in_cleanup = absl::MakeCleanup([&model_in]() { model_in.close(); });
    std::string model_data((std::istreambuf_iterator<char>(model_in)), std::istreambuf_iterator<char>());

    engine_ = TRTEngineFactory::instance()->runtime()->deserializeCudaEngine(
      model_data.data(), model_data.size(), nullptr
    );  // NOLINT
  }

  context_ = engine_->createExecutionContext();
}

void TRTEngine::build() {
}

void TRTEngine::set_session_options() {
}

void TRTEngine::create_session() {
}

void TRTEngine::sub_init() {
}

void TRTEngine::get_input_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *input_shapes
) {
  if (nullptr == input_shapes) {
    std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Input shapes is nullptr";
    throw std::runtime_error(err_msg);
  }
}

void TRTEngine::get_output_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *output_shapes
) {
  if (nullptr == output_shapes) {
    std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Output shapes is nullptr";
    throw std::runtime_error(err_msg);
  }
}

std::unique_ptr<TRTEngineFactory> TRTEngineFactory::instance_ = nullptr;
std::unique_ptr<nvinfer1::IRuntime> TRTEngineFactory::runtime_ = nullptr;

EngineFactory *TRTEngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new TRTEngineFactory());
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  }
  return instance_.get();
}

}  // namespace computational_advertising
