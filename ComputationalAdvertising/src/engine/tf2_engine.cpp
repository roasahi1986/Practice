// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/tf2_engine.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace computational_advertising {

TF2Engine::TF2Engine(const EngineConf& engine_conf) noexcept(false) :
  Engine(engine_conf),
  // engine_mtx_(),
  tags_(),
  session_opts_(),
  run_opts_(),
  model_bundle_() {
}

TF2Engine::~TF2Engine() {
  try {
    // std::unique_lock<std::shared_mutex> engine_lock(engine_mtx_);
    inited_ = false;
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  } catch (...) {
    LOG(ERROR) << "Unknown exception";
  }
}

std::string TF2Engine::brand() noexcept {
  return kBrandTF2;
}

void TF2Engine::warmup(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }
  infer(instance, score);
}

void TF2Engine::infer(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
  instance_to_tensor(*instance, &input_tensors);

  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status status = model_bundle_.GetSession()->Run(input_tensors, conf_.output_nodes, {}, &outputs);
  if (!status.ok()) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to run session: " + status.ToString();
    throw std::runtime_error(err_msg);
  }

  score_from_tensor(outputs, score);
}

void TF2Engine::trace(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }
  infer(instance, score);
}

void TF2Engine::get_input_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *input_shapes
) {
  if (nullptr == input_shapes) {
    std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Input shapes is nullptr";
    throw std::runtime_error(err_msg);
  }

  for (const auto& tensor_info : tf_model_meta_.input_metas) {
    (*input_shapes)[tensor_info.first] = tensor_info.second.shape;
  }
}

void TF2Engine::get_output_name_and_shape(
  absl::flat_hash_map<std::string, std::vector<int64_t>> *output_shapes
) {
  if (nullptr == output_shapes) {
    std::string err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Output shapes is nullptr";
    throw std::runtime_error(err_msg);
  }

  for (const auto& tensor_info : tf_model_meta_.output_metas) {
    (*output_shapes)[tensor_info.first] = tensor_info.second.shape;
  }
}

void TF2Engine::load() {
}

void TF2Engine::build() {
}

void TF2Engine::set_session_options() {
  tags_.insert(tensorflow::kSavedModelTagServe);
  // session_opts_.target = "local";

  tensorflow::OptimizerOptions tf_optimizer_opts;
  tf_optimizer_opts.set_do_constant_folding(true);
  tf_optimizer_opts.set_do_function_inlining(true);
  if (0 == conf_.opt_level) {
    tf_optimizer_opts.set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  } else {
    tf_optimizer_opts.set_opt_level(tensorflow::OptimizerOptions_Level_L1);
  }
  if (0 == conf_.jit_level) {
    tf_optimizer_opts.set_cpu_global_jit(false);
    tf_optimizer_opts.set_global_jit_level(tensorflow::OptimizerOptions_GlobalJitLevel_OFF);
  } else if (1 == conf_.jit_level) {
    tf_optimizer_opts.set_cpu_global_jit(true);
    tf_optimizer_opts.set_global_jit_level(tensorflow::OptimizerOptions_GlobalJitLevel_ON_1);
  } else {
    tf_optimizer_opts.set_cpu_global_jit(true);
    tf_optimizer_opts.set_global_jit_level(tensorflow::OptimizerOptions_GlobalJitLevel_ON_2);
  }

  session_opts_.config.mutable_graph_options()->mutable_optimizer_options()->CopyFrom(tf_optimizer_opts);
  session_opts_.config.set_intra_op_parallelism_threads(conf_.intra_op_parallelism_threads);
  session_opts_.config.set_inter_op_parallelism_threads(conf_.inter_op_parallelism_threads);
  set_gpu(&(session_opts_.config));
}

void TF2Engine::set_gpu(tensorflow::ConfigProto *tf_session_conf) noexcept(false) {
  if (nullptr == tf_session_conf) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
      + "Session config is nullptr";
    throw std::runtime_error(err_msg);
  }

  (*(tf_session_conf->mutable_device_count()))["GPU"] = 0;
}

void TF2Engine::create_session() {
  LOG(INFO) << "Loading saved model from: " << conf_.graph_file_loc;
  tensorflow::Status status = tensorflow::LoadSavedModel(
    session_opts_, run_opts_, conf_.graph_file_loc, tags_, &model_bundle_
  );  // NOLINT
  LOG(INFO) << "LoadSavedModel status: " << status.ToString();
}

void TF2Engine::sub_init() {
  for (const auto& tensor_name : conf_.input_nodes) {
    get_tf_tensor_meta_by_tf_operation_name(tensor_name, &tf_model_meta_.input_metas);
  }

  for (const auto& tensor_name : conf_.output_nodes) {
    get_tf_tensor_meta_by_tf_operation_name(tensor_name, &tf_model_meta_.output_metas);
  }

  LOG(INFO) << tf_model_meta_.to_string();

  const tensorflow::GraphDef& graph_def = model_bundle_.meta_graph_def.graph_def();
  for (int32_t i = 0; i < static_cast<int32_t>(graph_def.node_size()); ++i) {
    const tensorflow::NodeDef& node = graph_def.node(i);
    DLOG(INFO) << "node: " << node.name() << " [op: " << node.op() << "] is placed on device: " << node.device();
  }
}

// Get TFTensorMeta by TF_Operation name
void TF2Engine::get_tf_tensor_meta_by_tf_operation_name(
  const std::string& tf_operation_name,
  absl::flat_hash_map<std::string, TF2TensorMeta> *tf_tensor_meta
) {
  const tensorflow::GraphDef& graph_def = model_bundle_.meta_graph_def.graph_def();
  for (int32_t i = 0; i < static_cast<int32_t>(graph_def.node_size()); ++i) {
    const tensorflow::NodeDef& node = graph_def.node(i);
    if (node.name() != tf_operation_name) {
      continue;
    }
    TF2TensorMeta tensor_meta;
    tensor_meta.operation_name = node.name();
    tensor_meta.operation_type = node.op();
    tensor_meta.device         = node.device();

    // print all keys in node.attr()
    for (const auto& entry : node.attr()) {
      DLOG(INFO) << "name: " << node.name() << ", key: " << entry.first << ", value: " << entry.second.DebugString();
    }

    if (node.attr().find("shape") != node.attr().end()) {
      const tensorflow::AttrValue& shape_attr = node.attr().at("shape");
      tensor_meta.num_dims = shape_attr.shape().dim_size();
      tensor_meta.shape.clear();
      for (int32_t j = 0; j < static_cast<int32_t>(shape_attr.shape().dim_size()); ++j) {
        tensor_meta.shape.push_back(shape_attr.shape().dim(j).size());
      }
    } else if (node.attr().find("_output_shapes") != node.attr().end()
      && node.attr().at("_output_shapes").list().shape_size() > 0) {
      const tensorflow::AttrValue& shape_attr = node.attr().at("_output_shapes");
      tensor_meta.num_dims = shape_attr.list().shape(0).dim_size();
      tensor_meta.shape.clear();
      for (int32_t j = 0; j < static_cast<int32_t>(shape_attr.list().shape(0).dim_size()); ++j) {
        tensor_meta.shape.push_back(shape_attr.list().shape(0).dim(j).size());
      }
    }

    (*tf_tensor_meta)[tf_operation_name] = tensor_meta;
    return;
  }
  return;
}

void TF2Engine::instance_to_tensor(
  const Instance& instance, std::vector<std::pair<std::string, tensorflow::Tensor>> *input_tensors
) {
  absl::flat_hash_map<std::string, std::vector<int64_t>> input_shapes;
  get_input_name_and_shape(&input_shapes);

  for (const auto& feature_tensor : instance.features) {
    // Find the shape for this feature tensor based on its name
    auto it = input_shapes.find(feature_tensor.name);
    if (it == input_shapes.end()) {
      const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
        + "Shape for input tensor " + feature_tensor.name + " not found";
      throw std::runtime_error(err_msg);
    }
    std::vector<int64_t>& tensor_shape = it->second;
    tensor_shape[0] = feature_tensor.batch_size;

    // Create a TensorFlow tensor with the correct shape
    tensorflow::TensorShape tf_tensor_shape(tensor_shape);
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tf_tensor_shape);

    // Directly copy data into the tensor using Eigen::TensorMap for zero-copy assignment
    auto eigen_tensor = input_tensor.flat<float>();
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> eigen_tensor_map(
      eigen_tensor.data(), eigen_tensor.size()
    );  // NOLINT
    eigen_tensor_map = Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
      feature_tensor.data.data(), feature_tensor.data.size()
    );  // NOLINT

    input_tensors->emplace_back(feature_tensor.name, input_tensor);
  }
}

void TF2Engine::score_from_tensor(
  const std::vector<tensorflow::Tensor>& output_tensors, Score *score
) {
  absl::flat_hash_map<std::string, std::vector<int64_t>> output_shapes;
  get_output_name_and_shape(&output_shapes);

  score->targets.clear();
  int32_t i = 0;
  for (const auto& output_tensor : output_tensors) {
    const std::string& tensor_name = conf_.output_nodes[i++];

    Tensor target;
    target.name = tensor_name;
    target.batch_size = output_tensor.dim_size(0);  // Assuming the first dimension is batch size.

    // Directly assign data to target.data
    auto flat_tensor = output_tensor.flat<float>();
    target.data.assign(flat_tensor.data(), flat_tensor.data() + flat_tensor.size());

    score->targets.push_back(target);
  }
}

std::string TF2TensorMeta::to_string() {
  std::string message;
  absl::StrAppendFormat(&message,
    "  operation_name: %s\n  operation_type: %s\n  device: %s\n  num_dims: %d\n  shape: %s\n",
    operation_name.c_str(), operation_type.c_str(), device.c_str(), num_dims, absl::StrJoin(shape, ", ").c_str()
  );  // NOLINT

  return message;
}

std::string TF2ModelMeta::to_string() {
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

std::unique_ptr<TF2EngineFactory> TF2EngineFactory::instance_ = nullptr;
EngineFactory *TF2EngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new TF2EngineFactory());
  }
  return instance_.get();
}

}  // namespace computational_advertising
