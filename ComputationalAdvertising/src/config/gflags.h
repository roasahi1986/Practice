// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_CONFIG_GFLAGS_H_
#define COMPUTATIONALADVERTISING_SRC_CONFIG_GFLAGS_H_

#include <stdint.h>
#include <string>
#include "absl/flags/flag.h"
#include "absl/flags/declare.h"

ABSL_DECLARE_FLAG(std::string, host);
ABSL_DECLARE_FLAG(int32_t, port);

ABSL_DECLARE_FLAG(int32_t, number_of_inference_workers);
ABSL_DECLARE_FLAG(int32_t, batch_size);

ABSL_DECLARE_FLAG(std::string, local_model_dir);
ABSL_DECLARE_FLAG(int32_t, number_of_test_cases);
ABSL_DECLARE_FLAG(int32_t, number_of_producers);
ABSL_DECLARE_FLAG(int32_t, number_of_consumers);

ABSL_DECLARE_FLAG(int32_t, engine_opt_level);
ABSL_DECLARE_FLAG(int32_t, engine_jit_level);
ABSL_DECLARE_FLAG(int32_t, engine_inter_op_parallelism_threads);
ABSL_DECLARE_FLAG(int32_t, engine_intra_op_parallelism_threads);
ABSL_DECLARE_FLAG(bool, engin_use_global_thread_pool);
ABSL_DECLARE_FLAG(bool, engine_ort_parrallel_execution);

#endif  // COMPUTATIONALADVERTISING_SRC_CONFIG_GFLAGS_H_
