// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_POPULATION_ROSTER_H_
#define COMPUTATIONALADVERTISING_SRC_POPULATION_ROSTER_H_

#include <vector>
#include <string>
#include "absl/container/flat_hash_map.h"

namespace computational_advertising {

struct IndivadualInfo {
  std::string name;
  std::string age;
  std::string home_path;

  std::string graph_file_loc() const noexcept(false);
  std::string model_conf_loc() const noexcept(false);
};

struct Roster {
  absl::flat_hash_map<std::string, IndivadualInfo> indivaduals;
  void load(const std::string& path) noexcept(false);
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_POPULATION_ROSTER_H_
