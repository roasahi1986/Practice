// Copyright (C) 2021 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/process/process_status.h"
#include <atomic>
#include <future> // NOLINT
#include <thread> // NOLINT
#include "ComputationalAdvertising/src/util/os/resource_used.h"

namespace computational_advertising {

ProcessStatus::ProcessStatus() :
  cpu_used_(0.0),
  mem_used_(0.0),
  cpustamp_(0.0),
  timestamp_(0),
  finish_(),
  update_thread_(&ProcessStatus::update, this) {
}

ProcessStatus::~ProcessStatus() {
  finish_.set_value(true);
  if (update_thread_.joinable()) {
    update_thread_.join();
  }
}

void ProcessStatus::update() {
  struct ResourceUsed used;
  auto handle = finish_.get_future();
  while (handle.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
    struct timespec curr;

    clock_gettime(CLOCK_REALTIME, &curr);
    if (!get_process_resource_used(&used)) {
      continue;
    }

    const int64_t&& ts_prev = timestamp_.load();
    const int64_t&& ts_curr = curr.tv_sec * 1000000000 + curr.tv_nsec;
    const double&& elapse_sec = static_cast<double>((ts_curr - ts_prev) / 1000000000.0);

    const double&& cpu_prev = cpustamp_.load();
    const double&& cpu_curr = used.user_time + used.system_time;
    const double&& cpu_used = (cpu_curr - cpu_prev) / elapse_sec;

    cpu_used_.store(cpu_used);
    mem_used_.store(used.resident_mb);
    cpustamp_.store(cpu_curr);
    timestamp_.store(ts_curr);
  }
}

}  // namespace computational_advertising
