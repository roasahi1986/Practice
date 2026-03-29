// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_COMM_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_COMM_H_

#include <string>

namespace computational_advertising {

bool execute(const std::string& cmd);

bool execute(
  const std::string& cmd,
  std::string *stdo,
  bool include_stderr = false
); // NOLINT

bool execute(
  const std::string& cmd,
  std::string *stdo,
  std::string *stde
); // NOLINT

bool execute_vfork(
  const std::string& cmd,
  std::string *res = nullptr,
  int32_t buffer_size = 0
); // NOLINT

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_COMM_H_
