// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_OS_THREAD_CPU_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_OS_THREAD_CPU_H_

#include <stdint.h>

bool get_physical_cpu_num(int32_t *num);
bool get_thread_cpu(int64_t *cpu_id);
bool set_thread_cpu(int64_t cpu_id);

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_OS_THREAD_CPU_H_
