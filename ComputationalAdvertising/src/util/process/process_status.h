// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_PROCESS_PROCESS_STATUS_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_PROCESS_PROCESS_STATUS_H_

#include <atomic>
#include <future> // NOLINT
#include <thread> // NOLINT
#include "ComputationalAdvertising/src/util/os/resource_used.h"

namespace computational_advertising {

class ProcessStatus {
 public:
  ProcessStatus();
  ~ProcessStatus();

  ProcessStatus& operator=(const ProcessStatus&) = delete;
  ProcessStatus(const ProcessStatus&) = delete;

  double get_cpu_used() {
    return cpu_used_.load();
  }
  double get_mem_used() {
    return mem_used_.load();
  }

 private:
  void update();

  std::atomic<double> cpu_used_;
  std::atomic<double> mem_used_;
  std::atomic<double> cpustamp_;
  std::atomic<int64_t> timestamp_;

  std::promise<bool> finish_;
  std::thread update_thread_;
};  // class ProcessStatus

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_PROCESS_PROCESS_STATUS_H_
