// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/onnx_engine.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace computational_advertising {

ONNXEngine::ONNXEngine(const EngineConf& engine_conf) noexcept(false) :
  Engine(engine_conf),
  // engine_mtx_(),
  session_opts_(nullptr),
  env_(nullptr),
  session_(nullptr) {
}

ONNXEngine::~ONNXEngine() {
  try {
    // std::unique_lock<std::shared_mutex> engine_lock(engine_mtx_);
    inited_ = false;

    if (nullptr != session_) {
      delete session_;
      session_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Session deleted";
    }

    if (nullptr != env_) {
      delete env_;
      env_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Env deleted";
    }

    if (nullptr != session_opts_) {
      delete session_opts_;
      session_opts_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Session options deleted";
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  } catch (...) {
    LOG(ERROR) << "Unknown exception";
  }
}

std::string ONNXEngine::brand() noexcept {
  return kBrandONNX;
}

  // Perform inference using the ONNX runtime
void ONNXEngine::infer(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  run_session(instance, score, session_);
}

void ONNXEngine::run_session(Instance *instance, Score *score, Ort::Session *session) noexcept(false) {
  // Create memory info
  Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(
    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
  );  // NOLINT

  // Prepare input tensors
  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;

  // Set input data to the input tensors
  for (auto& feature : instance->features) {
    const std::string& feature_name = feature.name + ":0";
    const auto it = onnx_model_meta_.input_metas.find(feature_name);
    if (onnx_model_meta_.input_metas.end() != it) {
      std::vector<int64_t> tensor_shape = it->second.shape;
      tensor_shape[0] = feature.batch_size;
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
        info, feature.data.data(), feature.data.size(), tensor_shape.data(), tensor_shape.size()
      ));  // NOLINT
      input_names.push_back(it->second.name.c_str());
    }
  }

  // Prepare output tensors
  std::vector<const char*> output_names;
  std::vector<Ort::Value> output_tensors;

  // Set output data to the output tensors
  if (score->targets.size() != onnx_model_meta_.output_metas.size()) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Output size mismatch";
    throw std::runtime_error(err_msg);
  }
  for (auto& target : score->targets) {
    const std::string& target_name = target.name + ":0";
    const auto it = onnx_model_meta_.output_metas.find(target_name);
    if (onnx_model_meta_.output_metas.end() == it) {
      // std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      //   + conf_.brief() + "] " + "Output name not found: " + target_name;
      // throw std::runtime_error(err_msg);
      continue;
    }

    std::vector<int64_t> tensor_shape = it->second.shape;
    tensor_shape[0] = target.batch_size;
    target.data.resize(it->second.instance_size * target.batch_size);
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
      info, target.data.data(), target.data.size(), tensor_shape.data(), tensor_shape.size()
    ));  // NOLINT
    output_names.push_back(it->second.name.c_str());
  }

  // Run inference using the ONNX runtime
  Ort::RunOptions run_options;
  session->Run(
    run_options,
    input_names.data(), input_tensors.data(), input_names.size(),
    output_names.data(), output_tensors.data(), output_names.size()
  );  // NOLINT
}

void ONNXEngine::trace(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  std::unique_ptr<Ort::SessionOptions> session_opts(new Ort::SessionOptions());
  session_opts->EnableProfiling(conf_.name.c_str());
  session_opts->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_opts->SetIntraOpNumThreads(conf_.intra_op_parallelism_threads);
  session_opts->SetInterOpNumThreads(conf_.inter_op_parallelism_threads);
  std::unique_ptr<Ort::Env> env(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, conf_.name.c_str()));
  std::unique_ptr<Ort::Session> session(new Ort::Session(*env, conf_.graph_file_loc.c_str(), *session_opts));

  run_session(instance, score, session.get());

  static std::mutex trace_data_mtx;
  static std::atomic<int> trace_data_index = 0;
  std::lock_guard<std::mutex> lock(trace_data_mtx);
  trace_data_index.fetch_add(1);
  Ort::AllocatorWithDefaultOptions allocator;
  auto profile_file = session->EndProfilingAllocated(allocator);
  if (std::string(profile_file.get()) != std::string()) {
      LOG(INFO) << "ONNX profiling file has dump to " << std::string(profile_file.get());
  }
}

void ONNXEngine::load() {
}

void ONNXEngine::build() {
  // build engine
}

void ONNXEngine::set_session_options() {
  // set session options

  session_opts_ = new Ort::SessionOptions();
  session_opts_->SetInterOpNumThreads(conf_.inter_op_parallelism_threads);
  session_opts_->SetIntraOpNumThreads(conf_.intra_op_parallelism_threads);
  session_opts_->EnableCpuMemArena();
  // session_opts_->EnableOrtCustomOps();
  if (0 == conf_.opt_level) {
    session_opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  } else {
    session_opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  }
  if (conf_.ort_parrallel_execution) {
    session_opts_->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  } else {
    session_opts_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  }
  if (conf_.use_global_thread_pool) {
    session_opts_->DisablePerSessionThreads();
  }

  LOG(INFO) << "[" << conf_.detail() << "] Session options set";
}

void ONNXEngine::create_session() {
  // create session
  if (conf_.use_global_thread_pool) {
    Ort::ThreadingOptions threading_opts;
    threading_opts.SetGlobalInterOpNumThreads(conf_.inter_op_parallelism_threads);
    threading_opts.SetGlobalIntraOpNumThreads(conf_.intra_op_parallelism_threads);
    threading_opts.SetGlobalSpinControl(1);
    env_ = new Ort::Env(&(*threading_opts), OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, conf_.name.c_str());
  } else {
    env_ = new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, conf_.name.c_str());
  }
  session_ = new Ort::Session(*env_, conf_.graph_file_loc.c_str(), *session_opts_);
  LOG(INFO) << "[" << conf_.brief() << "] Session created";
}

void ONNXEngine::sub_init() {
  Ort::AllocatorWithDefaultOptions allocator;

  // Get input nodes info
  size_t num_input_nodes = session_->GetInputCount();
  for (int32_t i = 0; i < static_cast<int32_t>(num_input_nodes); ++i) {
    std::string input_name = std::string(session_->GetInputNameAllocated(i, allocator).get());
    auto tensor_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

    size_t instance_size = 1;
    std::vector<int64_t> shape;
    for (const auto& dim : tensor_shape) {
      shape.push_back(dim);
      if (dim > 0) {
        instance_size *= dim;
      }
    }

    onnx_model_meta_.input_metas[input_name] = ONNXTensorMeta{
      .name          = input_name,
      .num_dims      = static_cast<int32_t>(tensor_shape.size()),
      .shape         = shape,
      .instance_size = instance_size,
      .index         = i
    };
  }

  // Get output nodes info
  size_t num_output_nodes = session_->GetOutputCount();
  for (int32_t i = 0; i < static_cast<int32_t>(num_output_nodes); ++i) {
    std::string output_name = std::string(session_->GetOutputNameAllocated(i, allocator).get());
    auto tensor_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

    size_t instance_size = 1;
    std::vector<int64_t> shape;
    for (const auto& dim : tensor_shape) {
      shape.push_back(dim);
      if (dim > 0) {
        instance_size *= dim;
      }
    }

    onnx_model_meta_.output_metas[output_name] = ONNXTensorMeta{
      .name          = output_name,
      .num_dims      = static_cast<int32_t>(tensor_shape.size()),
      .shape         = shape,
      .instance_size = instance_size,
      .index         = i
    };
  }

  LOG(INFO) << onnx_model_meta_.to_string();
}

void ONNXEngine::get_input_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *input_shapes
) {
  if (nullptr == input_shapes) {
    std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Input shapes is nullptr";
    throw std::runtime_error(err_msg);
  }

  for (const auto& tensor_info : onnx_model_meta_.input_metas) {
    std::string input_name = tensor_info.first.substr(0, tensor_info.first.find(":"));
    (*input_shapes)[input_name] = tensor_info.second.shape;
  }
}

void ONNXEngine::get_output_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *output_shapes
) {
  if (nullptr == output_shapes) {
    std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Output shapes is nullptr";
    throw std::runtime_error(err_msg);
  }

  for (const auto& tensor_info : onnx_model_meta_.output_metas) {
    std::string output_name = tensor_info.first.substr(0, tensor_info.first.find(":"));
    (*output_shapes)[output_name] = tensor_info.second.shape;
  }
}

std::string ONNXTensorMeta::to_string() {
  std::string message;
  absl::StrAppendFormat(&message,
    "  num_dims: %d\n  instance_size: %llu\n  index: %d\n  shape: %s",
    num_dims, instance_size, index, absl::StrJoin(shape, ", ").c_str()
  );  // NOLINT

  return message;
}

std::string ONNXModelMeta::to_string() {
  std::string message;
  absl::StrAppendFormat(&message, "\ninput_metas: ");
  for (auto& entry : input_metas) {
    absl::StrAppendFormat(&message, "\n %s:", entry.first.c_str());
    absl::StrAppendFormat(&message, "\n%s", entry.second.to_string().c_str());
  }

  absl::StrAppendFormat(&message, "\noutput_metas:");
  for (auto& entry : output_metas) {
    absl::StrAppendFormat(&message, "\n %s:", entry.first.c_str());
    absl::StrAppendFormat(&message, "\n%s", entry.second.to_string().c_str());
  }

  return message;
}

std::unique_ptr<ONNXEngineFactory> ONNXEngineFactory::instance_ = nullptr;
EngineFactory *ONNXEngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new ONNXEngineFactory());
  }
  return instance_.get();
}

}  // namespace computational_advertising
