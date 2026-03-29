// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/tvm_engine.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ComputationalAdvertising/src/engine/onnx_engine.h"

namespace computational_advertising {

TVMEngine::TVMEngine(const EngineConf& engine_conf) noexcept(false) :
  Engine(engine_conf),
  // engine_mtx_(),
  device_type_(0),
  device_id_(0),
  dtype_code_(0),
  dtype_bits_(0),
  dtype_lanes_(0),
  module_(),
  set_input_(),
  run_(),
  get_output_(),
  input_shapes_(),
  output_shapes_(),
  release_() {
}

TVMEngine::~TVMEngine() {
  try {
    // std::unique_lock<std::shared_mutex> engine_lock(engine_mtx_);
    // release_();
    inited_ = false;
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  } catch (...) {
    LOG(ERROR) << "Unknown exception";
  }
}

std::string TVMEngine::brand() noexcept {
  return kBrandTVM;
}

// Perform inference using the TVM runtime
void TVMEngine::infer(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  std::vector<DLTensor *> input_tensors;
  for (auto& feature : instance->features) {
    const auto it = input_shapes_.find(feature.name);
    if (input_shapes_.end() != it) {
      std::vector<int64_t> tensor_shape = it->second;
      if (tensor_shape[0] != feature.batch_size) {
        const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
          + conf_.brief() + "] " + "Batch size mismatch";
        throw std::runtime_error(err_msg);
      }
      input_tensors.push_back(nullptr);

      const std::string& feature_name = it->first + ":0";
      TVMArrayAlloc(
        it->second.data(), it->second.size(),
        dtype_code_, dtype_bits_, dtype_lanes_,
        device_type_, device_id_, &(input_tensors.back())
      );  // NOLINT
      TVMArrayCopyFromBytes(input_tensors.back(), feature.data.data(), feature.data.size());
      set_input_(feature_name, input_tensors.back());
    }
  }

  run_();

  std::vector<DLTensor *> output_tensors;
  for (auto& target : score->targets) {
    const auto it = output_shapes_.find(target.name);
    if (output_shapes_.end() == it) {
      // std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      //   + conf_.brief() + "] " + "Output name not found: " + target_name;
      // throw std::runtime_error(err_msg);
      continue;
    }

    output_tensors.push_back(nullptr);
    TVMArrayAlloc(
      it->second.data(), it->second.size(),
      dtype_code_, dtype_bits_, dtype_lanes_,
      device_type_, device_id_, &(output_tensors.back())
    );  // NOLINT

    int64_t output_size = 1;
    for (auto& dim : it->second) {
      output_size *= dim;
    }
    const std::string& target_name = it->first + ":0";
    target.data.resize(output_size);
    get_output_(output_tensors.size() - 1, output_tensors.back());
    TVMArrayCopyToBytes(output_tensors.back(), target.data.data(), target.data.size() * sizeof(float));
  }

  for (auto& tensor : input_tensors) {
    TVMArrayFree(tensor);
  }
  for (auto& tensor : output_tensors) {
    TVMArrayFree(tensor);
  }
}

void TVMEngine::trace(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }
  infer(instance, score);
}

void TVMEngine::load() {
  dtype_code_  = kDLFloat;
  dtype_bits_  = 32;
  dtype_lanes_ = 1;
  device_type_ = kDLCUDA;
  device_id_   = 0;

  const std::string& so_file     = conf_.graph_file_loc + "mod.so";
  const std::string& json_file   = conf_.graph_file_loc + "mod.json";
  const std::string& params_file = conf_.graph_file_loc + "mod.params";
  const std::string& onnx_file   = conf_.graph_file_loc + "graph_tvm.onnx";

  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(so_file);

  std::ifstream json_in(json_file, std::ios::in);
  auto json_in_cleanup = absl::MakeCleanup([&json_in]() { json_in.close(); });
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());

  std::ifstream params_in(params_file, std::ios::binary);
  auto json_out_cleanup = absl::MakeCleanup([&params_in]() { params_in.close(); });
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());

  module_ = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data, mod_dylib, device_type_, device_id_);

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();
  tvm::runtime::PackedFunc load_params = module_.GetFunction("load_params");
  load_params(params_arr);

  set_input_  = module_.GetFunction("set_input");
  run_        = module_.GetFunction("run");
  get_output_ = module_.GetFunction("get_output");
  release_    = module_.GetFunction("release");

  EngineConf onnx_engine_conf = conf_;
  onnx_engine_conf.graph_file_loc = onnx_file;
  std::unique_ptr<Engine> onnx_engine(ONNXEngineFactory::instance()->create(onnx_engine_conf));
  onnx_engine->get_input_name_and_shape(&input_shapes_);
  onnx_engine->get_output_name_and_shape(&output_shapes_);

  // print input shapes
  for (const auto& input : input_shapes_) {
    LOG(INFO) << "input name: " << input.first << ", shape: " << absl::StrJoin(input.second, ",");
  }
  // print output shapes
  for (const auto& output : output_shapes_) {
    LOG(INFO) << "output name: " << output.first << ", shape: " << absl::StrJoin(output.second, ",");
  }
}

void TVMEngine::build() {
  // build engine
}

void TVMEngine::set_session_options() {
  // set session options
}

void TVMEngine::create_session() {
}

void TVMEngine::sub_init() {
}

void TVMEngine::get_input_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *input_shapes
) {
  input_shapes->insert(input_shapes_.begin(), input_shapes_.end());
}

void TVMEngine::get_output_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *output_shapes
) {
  output_shapes->insert(output_shapes_.begin(), output_shapes_.end());
}

std::unique_ptr<TVMEngineFactory> TVMEngineFactory::instance_ = nullptr;
EngineFactory *TVMEngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new TVMEngineFactory());
  }
  return instance_.get();
}

}  // namespace computational_advertising
