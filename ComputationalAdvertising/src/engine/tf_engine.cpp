// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/engine/tf_engine.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace computational_advertising {

TFEngine::TFEngine(const EngineConf& engine_conf) noexcept(false) :
  Engine(engine_conf),
  // engine_mtx_(),
  graph_buffer_(nullptr),
  graph_(nullptr),
  session_opts_(nullptr),
  session_(nullptr),
  default_run_option_buf_(nullptr),
  warmup_run_option_buf_(nullptr) {
}

TFEngine::~TFEngine() {
  try {
    // std::unique_lock<std::shared_mutex> engine_lock(engine_mtx_);
    inited_ = false;

    if (nullptr != session_) {
      TF_Status *tf_status = TF_NewStatus();
      auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

      TF_CloseSession(session_, tf_status);
      if (TF_GetCode(tf_status) != TF_OK) {
        const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
          + conf_.brief() + "] " + "Failed to close session: " + std::string(TF_Message(tf_status));
        throw std::runtime_error(err_msg);
      }
      LOG(INFO) << "[" << conf_.brief() << "] Session closed";

      TF_DeleteSession(session_, tf_status);
      if (TF_GetCode(tf_status) != TF_OK) {
        const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
          + conf_.brief() + "] " + "Failed to delete session: " + std::string(TF_Message(tf_status));
        throw std::runtime_error(err_msg);
      }
      session_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Session deleted";
    }

    if (nullptr != session_opts_) {
      TF_DeleteSessionOptions(session_opts_);
      session_opts_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Session options deleted";
    }

    if (nullptr != graph_) {
      TF_DeleteGraph(graph_);
      graph_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Graph deleted";
    }

    if (nullptr != graph_buffer_) {
      TF_DeleteBuffer(graph_buffer_);
      graph_buffer_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Graph buffer deleted";
    }

    if (nullptr != default_run_option_buf_) {
      TF_DeleteBuffer(default_run_option_buf_);
      default_run_option_buf_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Default run option buffer deleted";
    }
    if (nullptr != warmup_run_option_buf_) {
      TF_DeleteBuffer(warmup_run_option_buf_);
      warmup_run_option_buf_ = nullptr;
      LOG(INFO) << "[" << conf_.brief() << "] Default run option buffer deleted";
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  } catch (...) {
    LOG(ERROR) << "Unknown exception";
  }
}

std::string TFEngine::brand() noexcept {
  return kBrandTF;
}

void TFEngine::instance_to_tensor(
  Instance *instance, std::vector<TF_Tensor*> *input_tensors
) {
  for (auto& feature : instance->features) {
    const std::string& feature_name = feature.name;
    const auto it = tf_model_meta_.input_metas.find(feature_name);
    if (tf_model_meta_.input_metas.end() != it) {
      const auto& tensor_num_dims  = it->second.num_dims;
      const auto& tensor_data_type = it->second.data_type;

      size_t tensor_data_size = it->second.instance_size * static_cast<size_t>(feature.batch_size);
      DLOG(INFO) << "index: " << it->second.index << ", tensor_data_size: " << tensor_data_size
                 << ", batch_size: " << feature.batch_size
                 << ", tensor_num_dims: " << tensor_num_dims
                 << ", feature_data_size: " << feature.data.size() * sizeof(feature.data[0]);
      if (feature.data.size() * sizeof(feature.data[0]) != tensor_data_size) {
        const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
          + conf_.brief() + "] " + "Feature data size mismatch: " + feature_name;
        throw std::runtime_error(err_msg);
      }
      std::vector<int64_t> tensor_shape = it->second.shape;
      tensor_shape[0] = feature.batch_size;
      void *feature_value = static_cast<void *>(feature.data.data());

      TF_Tensor *tf_tensor = TF_NewTensor(
        tensor_data_type, tensor_shape.data(), tensor_num_dims, feature_value, tensor_data_size,
        [](void*, size_t, void*) {}, nullptr
      );  // NOLINT
      (*input_tensors)[it->second.index] = tf_tensor;
    } else {
      DLOG(INFO) << "Feature not found: " << feature_name;
    }
  }
}

void TFEngine::score_from_tensor(
  const std::vector<TF_Tensor*>& output_tensors, Score *score
) {
  // Convert TF_Output and TF_Tensor to BatchScore
  // score->targets.resize(tf_model_meta_.output_specs.size());
  if (score->targets.size() != tf_model_meta_.output_specs.size()) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + absl::StrFormat(
        "Score target size mismatch: %d != %d", score->targets.size(), tf_model_meta_.output_specs.size()
    );  // NOLINT
    throw std::runtime_error(err_msg);
  }
  for (auto& target : score->targets) {
    auto it = tf_model_meta_.output_metas.find(target.name);
    if (tf_model_meta_.output_metas.end() == it) {
      continue;
      // const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      //   + conf_.brief() + "] " + "Output not found: " + target.name;
      // throw std::runtime_error(err_msg);
    }
    // assert(it->second.data_size == sizeof(*target.data.data())
    const size_t data_size = it->second.instance_size * target.batch_size;
    char *data = static_cast<char*>(TF_TensorData(output_tensors[it->second.index]));
    target.data.resize(data_size / it->second.data_size);
    memcpy(target.data.data(), data, data_size);
  }
}

void TFEngine::warmup(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  // Convert BatchInstance to TF_Output and TF_Tensor
  std::vector<TF_Tensor*> input_tensors;
  input_tensors.resize(tf_model_meta_.input_specs.size(), nullptr);
  auto input_tensors_cleanup = absl::MakeCleanup([&input_tensors]() {
    for (auto& input_tensor : input_tensors) {
      if (nullptr != input_tensor) {
        TF_DeleteTensor(input_tensor);
      }
    }
  });

  std::vector<TF_Tensor*> output_tensors;
  output_tensors.resize(tf_model_meta_.output_specs.size(), nullptr);
  auto output_tensor_cleanup = absl::MakeCleanup([&output_tensors]() {
    for (auto& output_tensor : output_tensors) {
      if (nullptr != output_tensor) {
        TF_DeleteTensor(output_tensor);
      }
    }
  });

  instance_to_tensor(instance, &input_tensors);
  run_session(&input_tensors, &output_tensors, warmup_run_option_buf_);
  score_from_tensor(output_tensors, score);
}

void TFEngine::infer(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  // Convert BatchInstance to TF_Output and TF_Tensor
  std::vector<TF_Tensor*> input_tensors;
  input_tensors.resize(tf_model_meta_.input_specs.size(), nullptr);
  auto input_tensors_cleanup = absl::MakeCleanup([&input_tensors]() {
    for (auto& input_tensor : input_tensors) {
      if (nullptr != input_tensor) {
        TF_DeleteTensor(input_tensor);
      }
    }
  });

  std::vector<TF_Tensor*> output_tensors;
  output_tensors.resize(tf_model_meta_.output_specs.size(), nullptr);
  auto output_tensor_cleanup = absl::MakeCleanup([&output_tensors]() {
    for (auto& output_tensor : output_tensors) {
      if (nullptr != output_tensor) {
        TF_DeleteTensor(output_tensor);
      }
    }
  });

  instance_to_tensor(instance, &input_tensors);
  run_session(&input_tensors, &output_tensors, default_run_option_buf_);
  score_from_tensor(output_tensors, score);
}

void TFEngine::trace(Instance *instance, Score *score) noexcept(false) {
  // std::shared_lock<std::shared_mutex> engine_lock(engine_mtx_);
  if (!inited_) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Engine not initialized";
    throw std::runtime_error(err_msg);
  }

  tensorflow::RunOptions tf_run_opts;
  tf_run_opts.set_trace_level(tensorflow::RunOptions_TraceLevel_FULL_TRACE);
  std::string tf_run_opts_str;
  tf_run_opts.SerializeToString(&tf_run_opts_str);
  TF_Buffer *tf_run_opts_data = TF_NewBufferFromString(
    reinterpret_cast<void*>(tf_run_opts_str.data()), tf_run_opts_str.size()
  );  // NOLINT
  auto tf_run_opts_data_cleanup = absl::MakeCleanup([&tf_run_opts_data]() { TF_DeleteBuffer(tf_run_opts_data); });

  TF_Buffer *tf_metadata = TF_NewBuffer();
  auto tf_metadata_cleanup = absl::MakeCleanup([&tf_metadata]() { TF_DeleteBuffer(tf_metadata); });

  std::vector<TF_Tensor*> input_tensors;
  input_tensors.resize(tf_model_meta_.input_specs.size(), nullptr);
  auto input_tensors_cleanup = absl::MakeCleanup([&input_tensors]() {
    for (auto& input_tensor : input_tensors) {
      if (nullptr != input_tensor) {
        TF_DeleteTensor(input_tensor);
      }
    }
  });

  std::vector<TF_Tensor*> output_tensors;
  output_tensors.resize(tf_model_meta_.output_specs.size(), nullptr);
  auto output_tensors_cleanup = absl::MakeCleanup([&output_tensors]() {
    for (auto& output_tensor : output_tensors) {
      if (nullptr != output_tensor) {
        TF_DeleteTensor(output_tensor);
      }
    }
  });

  instance_to_tensor(instance, &input_tensors);
  run_session(&input_tensors, &output_tensors, tf_run_opts_data, tf_metadata);
  score_from_tensor(output_tensors, score);

  static std::mutex trace_data_mtx;
  static std::atomic<int> trace_data_index = 0;
  std::lock_guard<std::mutex> lock(trace_data_mtx);
  trace_data_index.fetch_add(1);
  std::string trace_data_str(reinterpret_cast<const char*>(tf_metadata->data), tf_metadata->length);
  std::ofstream ofs("trace_data_" + std::to_string(trace_data_index.load()) + ".pb");
  ofs << trace_data_str;
  ofs.close();
}

void TFEngine::run_session(
  std::vector<TF_Tensor*> *input_tensors,
  std::vector<TF_Tensor*> *output_tensors,
  TF_Buffer *tf_run_opts,
  TF_Buffer *tf_metadata
) {
  TF_Status *tf_status = TF_NewStatus();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

  TF_SessionRun(
    session_, tf_run_opts,
    tf_model_meta_.input_specs.data(), input_tensors->data(), input_tensors->size(),
    tf_model_meta_.output_specs.data(), output_tensors->data(), output_tensors->size(),
    nullptr, 0, tf_metadata, tf_status
  );  // NOLINT

  if (TF_GetCode(tf_status) != TF_OK) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to run session: " + std::string(TF_Message(tf_status));
    throw std::runtime_error(err_msg);
  }
}

void TFEngine::load() {
  const std::string& graph_file = conf_.graph_file_loc;
  std::ifstream file(graph_file, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to open graph file: " + graph_file;
    throw std::runtime_error(err_msg);
  }
  auto file_cleanup = absl::MakeCleanup([&file]() { file.close(); });

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  char *buffer_data = new char[size];
  if (!file.read(buffer_data, size)) {
    delete[] buffer_data;

    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to read graph file: " + graph_file;
    throw std::runtime_error(err_msg);
  }

  graph_buffer_ = new TF_Buffer();
  graph_buffer_->data = buffer_data;
  graph_buffer_->length = size;
  graph_buffer_->data_deallocator = [](void *data, size_t length) {
    delete[] static_cast<char*>(data);
  };

  LOG(INFO) << "[" << conf_.brief() << "] Graph file loaded: " << graph_file;
}

void TFEngine::build() {
  TF_Status *tf_status = TF_NewStatus();
  TF_ImportGraphDefOptions *tf_import_graph_def_opts = TF_NewImportGraphDefOptions();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status, &tf_import_graph_def_opts]() {
    TF_DeleteStatus(tf_status);
    TF_DeleteImportGraphDefOptions(tf_import_graph_def_opts);
  });

  graph_ = TF_NewGraph();
  TF_GraphImportGraphDef(graph_, graph_buffer_, tf_import_graph_def_opts, tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to import graph: " + std::string(TF_Message(tf_status));
    throw std::runtime_error(err_msg);
  }

  LOG(INFO) << "[" << conf_.brief() << "] Graph imported";
}

void TFEngine::set_session_options() {
  // refer to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
  TF_Status *tf_status = TF_NewStatus();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

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

  tensorflow::ConfigProto tf_session_conf;
  tf_session_conf.mutable_graph_options()->mutable_optimizer_options()->CopyFrom(tf_optimizer_opts);
  tf_session_conf.set_intra_op_parallelism_threads(conf_.intra_op_parallelism_threads);
  tf_session_conf.set_inter_op_parallelism_threads(conf_.inter_op_parallelism_threads);
  set_gpu(&tf_session_conf);

  {
    tensorflow::ThreadPoolOptionProto *thread_pool_opt =
      tf_session_conf.mutable_session_inter_op_thread_pool()->Add();
    thread_pool_opt->set_num_threads(1);
    thread_pool_opt->set_global_name(std::string(kGlobalInterOpThreadPool) + "_warmup");
  }

  if (conf_.use_global_thread_pool) {
    tf_session_conf.set_use_per_session_threads(false);
    tensorflow::ThreadPoolOptionProto *thread_pool_opt =
      tf_session_conf.mutable_session_inter_op_thread_pool()->Add();
    thread_pool_opt->set_num_threads(conf_.inter_op_parallelism_threads);
    thread_pool_opt->set_global_name(
      std::string(kGlobalInterOpThreadPool) + "_"
      + std::to_string(conf_.inter_op_parallelism_threads)
    );  // NOLINT
  } else {
    tf_session_conf.set_use_per_session_threads(true);
  }

  // tf_session_conf.mutable_experimental()->set_collective_deterministic_sequential_execution(true);
  // tf_session_conf.mutable_experimental()->set_collective_nccl(true);
  // tf_session_conf.mutable_experimental()->set_disable_thread_spinning(true);
  // tf_session_conf.mutable_experimental()->set_optimize_for_static_graph(true);
  // tf_session_conf.mutable_experimental()->set_use_tfrt(true);
  // tf_session_conf.mutable_experimental()->set_xla_fusion_autotuner_thresh(256);
  LOG(INFO) << "[" << conf_.detail() << "] Session config:\n" << tf_session_conf.DebugString();

  std::string tf_session_conf_str;
  tf_session_conf.SerializeToString(&tf_session_conf_str);

  session_opts_ = TF_NewSessionOptions();
  TF_SetConfig(session_opts_, tf_session_conf_str.data(), tf_session_conf_str.size(), tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to set session config: " + std::string(TF_Message(tf_status));
    throw std::runtime_error(err_msg);
  }

  tensorflow::RunOptions default_run_opt;
  default_run_opt.mutable_experimental()->set_use_run_handler_pool(true);
  default_run_opt.mutable_experimental()->mutable_run_handler_pool_options()->set_priority(1);
  if (conf_.use_global_thread_pool) {
    default_run_opt.set_inter_op_thread_pool(1);
  }
  std::string default_run_opt_str;
  default_run_opt.SerializeToString(&default_run_opt_str);
  default_run_option_buf_ = TF_NewBufferFromString(
    static_cast<void*>(default_run_opt_str.data()), default_run_opt_str.size()
  );  // NOLINT

  LOG(INFO) << "[" << conf_.brief() << "] Session options set";

  default_run_opt.set_inter_op_thread_pool(0);
  default_run_opt.SerializeToString(&default_run_opt_str);
  default_run_opt.mutable_experimental()->set_use_run_handler_pool(false);
  warmup_run_option_buf_ = TF_NewBufferFromString(
    static_cast<void*>(default_run_opt_str.data()), default_run_opt_str.size()
  );  // NOLINT
}

void TFEngine::set_gpu(tensorflow::ConfigProto *tf_session_conf) noexcept(false) {
  if (nullptr == tf_session_conf) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
      + "Session config is nullptr";
    throw std::runtime_error(err_msg);
  }

  (*(tf_session_conf->mutable_device_count()))["GPU"] = 0;
  // (*(tf_session_conf->mutable_device_count()))["CPU"] = 1;
}

void TFEngine::create_session() {
  TF_Status *tf_status = TF_NewStatus();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

  session_ = TF_NewSession(graph_, session_opts_, tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to create session: " + std::string(TF_Message(tf_status));
    throw std::runtime_error(err_msg);
  }

  LOG(INFO) << "[" << conf_.brief() << "] Session created";
}

void TFEngine::sub_init() {
  TF_Status *tf_status = TF_NewStatus();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

  int32_t input_index = 0;
  for (const auto& tensor_name : conf_.input_nodes) {
    get_tf_tensor_meta_by_tf_operation_name(tensor_name, &tf_model_meta_.input_metas, &input_index);
  }

  int32_t output_index = 0;
  for (const auto& tensor_name : conf_.output_nodes) {
    get_tf_tensor_meta_by_tf_operation_name(tensor_name, &tf_model_meta_.output_metas, &output_index);
  }

  tf_model_meta_.input_specs.resize(tf_model_meta_.input_metas.size());
  for (const auto& tensor_info : tf_model_meta_.input_metas) {
    tf_model_meta_.input_specs[tensor_info.second.index] = (*(tensor_info.second.output));
  }
  tf_model_meta_.output_specs.resize(tf_model_meta_.output_metas.size());
  for (const auto& tensor_info : tf_model_meta_.output_metas) {
    tf_model_meta_.output_specs[tensor_info.second.index] = (*(tensor_info.second.output));
  }

  LOG(INFO) << tf_model_meta_.to_string();
}

// Iterate through the operations of a graph
void TFEngine::iterate_through_operations(std::function<void(TF_Operation*)> do_something_with_operation) {
  size_t pos = 0;
  TF_Operation *oper;
  while ((oper = TF_GraphNextOperation(graph_, &pos)) != nullptr) {
    do_something_with_operation(oper);
  }
}

// Get TFTensorMeta by TF_Operation name
void TFEngine::get_tf_tensor_meta_by_tf_operation_name(
  const std::string& tf_operation_name,
  absl::flat_hash_map<std::string, TFTensorMeta> *tf_tensor_meta,
  int32_t *index
) {
  TF_Status *tf_status = TF_NewStatus();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

  TF_Operation *tf_operation = TF_GraphOperationByName(graph_, tf_operation_name.c_str());
  if (nullptr == tf_operation) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to get operation: " + tf_operation_name;
    throw std::runtime_error(err_msg);
  }

  int32_t op_num_outputs = TF_OperationNumOutputs(tf_operation);
  if (op_num_outputs > 1) {
    for (int32_t i = 0; i < op_num_outputs; ++i) {
      std::string name = tf_operation_name + ":" + std::to_string(i);
      (*tf_tensor_meta)[name] = TFTensorMeta();
      convert_tf_output_to_tf_tensor_meta(TF_Output {tf_operation, i}, &(*tf_tensor_meta)[name]);
      if (nullptr != index) {
        (*tf_tensor_meta)[name].index = (*index)++;
      }
    }
  } else {
      (*tf_tensor_meta)[tf_operation_name] = TFTensorMeta();
      convert_tf_output_to_tf_tensor_meta(TF_Output {tf_operation, 0}, &(*tf_tensor_meta)[tf_operation_name]);
      if (nullptr != index) {
        (*tf_tensor_meta)[tf_operation_name].index = (*index)++;
      }
  }
}

// Convert TF_Output to TFTensorMeta
void TFEngine::convert_tf_output_to_tf_tensor_meta(const TF_Output& tf_output, TFTensorMeta *tf_tensor_meta) {
  TF_Status *tf_status = TF_NewStatus();
  auto tf_status_cleanup = absl::MakeCleanup([&tf_status]() { TF_DeleteStatus(tf_status); });

  tf_tensor_meta->output = new TF_Output(tf_output);
  tf_tensor_meta->operation_name = TF_OperationName(tf_output.oper);
  tf_tensor_meta->operation_type = TF_OperationOpType(tf_output.oper);
  tf_tensor_meta->operation_num_inputs = TF_OperationNumInputs(tf_output.oper);
  tf_tensor_meta->operation_num_outputs = TF_OperationNumOutputs(tf_output.oper);
  tf_tensor_meta->data_type = TF_OperationOutputType(tf_output);
  tf_tensor_meta->data_size = TF_DataTypeSize(tf_tensor_meta->data_type);
  tf_tensor_meta->num_dims = TF_GraphGetTensorNumDims(graph_, tf_output, tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to get tensor num dims: " + std::string(TF_Message(tf_status));
    throw std::runtime_error(err_msg);
  }
  if (!(tf_tensor_meta->num_dims > 0)) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + tf_tensor_meta->operation_name + " has no dimensions";
    throw std::runtime_error(err_msg);
  }

  tf_tensor_meta->shape.resize(tf_tensor_meta->num_dims);
  TF_GraphGetTensorShape(graph_, tf_output, tf_tensor_meta->shape.data(), tf_tensor_meta->num_dims, tf_status);
  if (TF_GetCode(tf_status) != TF_OK) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + conf_.brief() + "] " + "Failed to get tensor shape: " + std::string(TF_Message(tf_status));
    throw std::runtime_error(err_msg);
  }

  tf_tensor_meta->instance_size = tf_tensor_meta->data_size;
  for (int32_t i = 1; i < tf_tensor_meta->num_dims; ++i) {
    tf_tensor_meta->instance_size *= tf_tensor_meta->shape[i];
  }
}

void TFEngine::get_input_name_and_shape(
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

void TFEngine::get_output_name_and_shape(
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

std::string TFTensorMeta::to_string() {
  std::string message;
  absl::StrAppendFormat(&message,
    "  operation_name: %s\n  operation_type: %s\n  operation_num_inputs: %d\n  operation_num_outputs: %d\n"
    "  data_type: %d\n  data_size: %d\n  num_dims: %d\n  instance_size: %llu\n  index: %d\n  shape: %s",
    operation_name.c_str(), operation_type.c_str(), operation_num_inputs, operation_num_outputs,
    data_type, data_size, num_dims, instance_size, index, absl::StrJoin(shape, ", ").c_str()
  );  // NOLINT

  return message;
}

std::string TFModelMeta::to_string() {
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

std::unique_ptr<TFEngineFactory> TFEngineFactory::instance_ = nullptr;
EngineFactory *TFEngineFactory::instance() {
  if (nullptr == instance_) {
    instance_.reset(new TFEngineFactory());
  }
  return instance_.get();
}

}  // namespace computational_advertising
