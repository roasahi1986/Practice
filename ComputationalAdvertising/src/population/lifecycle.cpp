// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/population/lifecycle.h"
#include <memory>
#include <string>
#include "ComputationalAdvertising/src/engine/tf_engine.h"
#include "ComputationalAdvertising/src/engine/onnx_engine.h"

namespace computational_advertising {

Lifecycle::Lifecycle(const IndivadualInfo& indivadual_info) noexcept(false) :
  indivadual_info_(indivadual_info) {
  model_meta_.load(indivadual_info_.model_conf_loc());

  engine_conf_.name = indivadual_info_.name;
  engine_conf_.version = indivadual_info_.age;
  engine_conf_.graph_file_loc = indivadual_info.graph_file_loc();
  // engine_conf_.input_nodes = ;
  // engine_conf_.output_nodes = ;
  // engine_conf_.opt_level;
  // engine_conf_.jit_level;
  // engine_conf_.inter_op_parallelism_threads;
  // engine_conf_.intra_op_parallelism_threads;
  engine_ = std::unique_ptr<TFEngine>(new TFEngine(engine_conf_));
}

Lifecycle::~Lifecycle() = default;

void Lifecycle::age(const std::string& new_age) noexcept(false) {}
void Lifecycle::undertake(Instance *instance, Score *score) noexcept(false) {
  engine_->infer(instance, score);
}

}  // namespace computational_advertising
