// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_POPULATION_LIFECYCLE_H_
#define COMPUTATIONALADVERTISING_SRC_POPULATION_LIFECYCLE_H_

#include <memory>
#include <vector>
#include <string>
#include "ComputationalAdvertising/src/engine/sample.h"
#include "ComputationalAdvertising/src/engine/engine.h"
#include "ComputationalAdvertising/src/embedding/embedding.h"
#include "ComputationalAdvertising/src/population/roster.h"
#include "ComputationalAdvertising/src/population/model_spec.h"

namespace computational_advertising {

class Lifecycle {
 public:
  explicit Lifecycle(const IndivadualInfo& indivadual_info) noexcept(false);
  virtual ~Lifecycle();

  Lifecycle& operator=(const Lifecycle&) = delete;
  Lifecycle(const Lifecycle&) = delete;

  void age(const std::string& new_age) noexcept(false);
  void undertake(Instance *instance, Score *score) noexcept(false);

 private:
  std::vector<std::string>   memories_;
  std::string                age_;
  IndivadualInfo             indivadual_info_;
  ModelMeta                  model_meta_;
  EngineConf                 engine_conf_;
  std::unique_ptr<Engine>    engine_;
  std::unique_ptr<Embedding> embedding_;
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_POPULATION_LIFECYCLE_H_
