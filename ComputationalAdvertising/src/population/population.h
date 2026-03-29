// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_POPULATION_POPULATION_H_
#define COMPUTATIONALADVERTISING_SRC_POPULATION_POPULATION_H_

#include <memory>
#include <mutex>  // NOLINT
#include <shared_mutex>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "ComputationalAdvertising/src/population/roster.h"
#include "ComputationalAdvertising/src/population/lifecycle.h"

namespace computational_advertising {

class Population {
 public:
  explicit Population(const std::string& settlement_path) noexcept;
  virtual ~Population();

  Population& operator=(const Population&) = delete;
  Population(const Population&) = delete;

  void evolve() noexcept(false);
  std::shared_ptr<Lifecycle> summon(const std::string& name) noexcept(false);

 private:
  void born(const std::string& name, const IndivadualInfo& indivadual_info) noexcept(false);
  void die(const std::string& name) noexcept(false);

  std::mutex evolvement_mutex_;
  std::shared_mutex population_mutex_;
  std::string settlement_path_;
  std::unique_ptr<Roster> roster_;
  absl::flat_hash_map<std::string, std::shared_ptr<Lifecycle>> indivaduals_;
};

}  // namespace computational_advertising
#endif  // COMPUTATIONALADVERTISING_SRC_POPULATION_POPULATION_H_
