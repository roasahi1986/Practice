// Copyright (C) 2023 lusyu1986@icloud.com

#include <stdint.h>
#include <exception>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "BShoshany/BS_thread_pool.hpp"

#include "ComputationalAdvertising/src/util/process/process_initiator.h"
#include "ComputationalAdvertising/src/util/functional/timer.h"
#include "ComputationalAdvertising/src/config/gflags.h"
#include "ComputationalAdvertising/src/engine/sample.h"
#include "ComputationalAdvertising/src/engine/engine.h"
#include "select_engine.h"  // NOLINT

computational_advertising::Engine *create_engine();

int main(int argc, char **argv) {
  computational_advertising::init(argc, argv);
  try {
    std::unique_ptr<computational_advertising::Engine> engine(create_demo_engine_3());
    if (nullptr == engine.get()) {
      LOG(ERROR) << "Failed to create engine";
      return -1;
    }

    computational_advertising::PerfIndex perf_index;
    engine->perf(
      absl::GetFlag(FLAGS_number_of_consumers),
      absl::GetFlag(FLAGS_number_of_test_cases),
      absl::GetFlag(FLAGS_batch_size), &perf_index
    );  //  NOLINT
    LOG(INFO) << "Summary:\n" << perf_index.DebugString();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  } catch (...) {
    LOG(ERROR) << "Unknown exception";
  }
  LOG(INFO) << "Done";

  return 0;
}
