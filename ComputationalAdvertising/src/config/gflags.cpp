// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/config/gflags.h"

#include <string>

#include "absl/flags/flag.h"

ABSL_FLAG(std::string, host, "", "Host");
ABSL_FLAG(int32_t, port, 8610, "Port");

ABSL_FLAG(int32_t, number_of_inference_workers, 16, "The number of inference workers");
ABSL_FLAG(int32_t, batch_size, 128, "Batch size");

ABSL_FLAG(std::string, local_model_dir, "", "Local model directory");
ABSL_FLAG(int32_t, number_of_test_cases, 1000, "Then number of test cases");
ABSL_FLAG(int32_t, number_of_producers, 1, "The number of producers");
ABSL_FLAG(int32_t, number_of_consumers, 1, "The number of consumers");

ABSL_FLAG(int32_t, engine_opt_level, 1, "Optimization level");
ABSL_FLAG(int32_t, engine_jit_level, 0, "JIT level");
ABSL_FLAG(int32_t, engine_inter_op_parallelism_threads, 16, "Inter op parallelism threads");
ABSL_FLAG(int32_t, engine_intra_op_parallelism_threads, 16, "Intra op parallelism threads");
ABSL_FLAG(bool, engin_use_global_thread_pool, true, "Use global thread pool");
ABSL_FLAG(bool, engine_ort_parrallel_execution, false, "ORT parallel execution");
