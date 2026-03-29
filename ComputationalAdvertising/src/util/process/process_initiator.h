// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_PROCESS_PROCESS_INITIATOR_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_PROCESS_PROCESS_INITIATOR_H_

#include <signal.h>
#include <stdlib.h>
#include <execinfo.h>
#include <functional>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/debugging/symbolize.h"
#include "absl/debugging/failure_signal_handler.h"

namespace computational_advertising {

const std::vector<int32_t> kExitSignals = {
  SIGTERM, SIGSEGV, SIGINT, SIGILL, SIGABRT, SIGFPE
};  // NOLINT

void default_handler(int32_t signal) {
  LOG(ERROR) << "process terminated by signal: " << signal;
  exit(EXIT_FAILURE);
}

void init(
  int32_t argc, char **argv, void (*signal_handler)(int32_t) = &default_handler
) {
  // google::AllowCommandLineReparsing();
  // google::ParseCommandLineFlags(&argc, &argv, true);
  absl::ParseCommandLine(argc, argv);

  // where the "nvvm/libdevice" directory located
  // for more configuration options, refer to:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/jit/flags.cc
  // https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc
  // https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/
  setenv("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8", 1);
  setenv("TF_XLA_FLAGS", "--tf_xla_cpu_global_jit", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("MKL_DYNAMIC", "FALSE", 1);
  setenv("OMP_DYNAMIC", "FALSE", 1);
  setenv("TF_GPU_ALLOCATOR", "cuda_malloc_async", 1);
  // setenv("TF_DISABLE_MKL", "1", 1);


  // FLAGS_logbufsecs = 0;
  // FLAGS_max_log_size = 1024;
  // FLAGS_minloglevel = google::INFO;
  // FLAGS_logtostdout = true;
  // google::InitGoogleLogging(argv[0]);

  for (auto& sig : kExitSignals) {
    signal(sig, signal_handler);
  }

  absl::InitializeSymbolizer(argv[0]);
  absl::FailureSignalHandlerOptions signal_handler_options;
  signal_handler_options.call_previous_handler = true;
  absl::InstallFailureSignalHandler(signal_handler_options);
}

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_PROCESS_PROCESS_INITIATOR_H_
