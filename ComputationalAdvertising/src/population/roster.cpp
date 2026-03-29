// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/population/roster.h"

#include <string>

namespace computational_advertising {

std::string IndivadualInfo::graph_file_loc() const noexcept(false) {
  return home_path + "/" + name + "/" + age + "/graph";
}

std::string IndivadualInfo::model_conf_loc() const noexcept(false) {
  return home_path + "/" + name + "/model_conf.json";
}

void Roster::load(const std::string& path) noexcept(false) {}

}  // namespace computational_advertising
