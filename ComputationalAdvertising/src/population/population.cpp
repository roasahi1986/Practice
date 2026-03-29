// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/population/population.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "BShoshany/BS_thread_pool.hpp"

namespace computational_advertising {

static const char kPopulationConfFileName[] = "__list__.json";
static const int32_t kEvolveThreadNum = 4;

Population::Population(const std::string& settlement_path) noexcept :
  settlement_path_(settlement_path) {}

Population::~Population() {}

void Population::evolve() noexcept(false) {
  std::string pupulation_conf_file = settlement_path_ + "/" + kPopulationConfFileName;

  std::lock_guard lock(evolvement_mutex_);
  roster_->load(pupulation_conf_file);
  absl::flat_hash_map<std::string, std::shared_ptr<Lifecycle>> indivaduals;
  {
    std::shared_lock lock(population_mutex_);
    indivaduals = indivaduals_;
  }

  BS::thread_pool evolve_thread_pool(kEvolveThreadNum);

  for (auto& [name, lifecycle] : indivaduals) {
    auto indivadual_ptr = roster_->indivaduals.find(name);
    if (roster_->indivaduals.end() == indivadual_ptr) {
      evolve_thread_pool.push_task(
        [this](const std::string name) {
          try {
            this->die(name);
          } catch (const std::exception& e) {
            LOG(ERROR) << e.what();
          } catch (...) {
            LOG(ERROR) << "unknown exception";
          }
        }, name
      );  // NOLINT
    } else {
      evolve_thread_pool.push_task(
        [](std::shared_ptr<Lifecycle> lifecycle, const std::string age) {
          try {
            lifecycle->age(age);
          } catch (const std::exception& e) {
            LOG(ERROR) << e.what();
          } catch (...) {
            LOG(ERROR) << "unknown exception";
          }
        }, lifecycle, roster_->indivaduals[name].age
      );  // NOLINT
    }
  }

  for (const auto& [name, indivadual_info] : roster_->indivaduals) {
    if (indivaduals_.find(name) == indivaduals_.end()) {
      evolve_thread_pool.push_task(
        [this](const std::string name, const IndivadualInfo indivadual_info) {
          try {
            this->born(name, indivadual_info);
          } catch (const std::exception& e) {
            LOG(ERROR) << e.what();
          } catch (...) {
            LOG(ERROR) << "unknown exception";
          }
        }, name, indivadual_info
      ); // NOLINT
    }
  }

  evolve_thread_pool.wait_for_tasks();
}

void Population::born(const std::string& name, const IndivadualInfo& indivadual_info) noexcept(false) {
  std::shared_ptr<Lifecycle> womb = std::make_shared<Lifecycle>(indivadual_info);
  {
    std::unique_lock lock(population_mutex_);
    indivaduals_.try_emplace(name, womb);
  }
}

void Population::die(const std::string& name) noexcept(false) {
  std::shared_ptr<Lifecycle> heaven = indivaduals_[name];
  {
    std::unique_lock lock(population_mutex_);
    indivaduals_.erase(name);
  }
}

std::shared_ptr<Lifecycle> Population::summon(const std::string& name) noexcept(false) {
  std::shared_lock lock(population_mutex_);

  auto lifecycle = indivaduals_.find(name);
  if (indivaduals_.end() == lifecycle) {
    return nullptr;
  }

  return lifecycle->second;
}

}  // namespace computational_advertising
