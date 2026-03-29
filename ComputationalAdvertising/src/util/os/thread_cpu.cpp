// Copyright (C) 2021 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/os/thread_cpu.h"

#ifdef __APPLE__

#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>

bool get_physical_cpu_num(int32_t *num) {
  if (nullptr == num) {
    return false;
  }
  size_t size = sizeof(*num);
  if (-1 == sysctlbyname("hw.physicalcpu", num, &size, nullptr, 0)) {
    return false;
  }
  return true;
}

bool get_thread_cpu(int64_t *cpu_id) {
  if (nullptr == cpu_id) {
    return false;
  }

  thread_port_t thread_port = pthread_mach_thread_np(pthread_self());

  struct thread_affinity_policy policy;
  policy.affinity_tag = 0;
  mach_msg_type_number_t count = THREAD_AFFINITY_POLICY_COUNT;
  boolean_t get_default = FALSE;

  kern_return_t ret = thread_policy_get(
    thread_port, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
    &count, &get_default
  );  // NOLINT
  if (KERN_SUCCESS != ret) {
    return false;
  }

  *cpu_id = policy.affinity_tag;
  return true;
}

bool set_thread_cpu(int64_t cpu_id) {
  thread_port_t thread_port = pthread_mach_thread_np(pthread_self());

  struct thread_affinity_policy policy;
  policy.affinity_tag = cpu_id;

  kern_return_t ret = thread_policy_set(
    thread_port, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
    THREAD_AFFINITY_POLICY_COUNT
  );  // NOLINT
  if (KERN_SUCCESS != ret) {
    return false;
  }

  return true;
}

#elif defined(__linux__)

#include <unistd.h>
#include <pthread.h>
#include <iostream>

bool get_physical_cpu_num(int32_t *num) {
  if (nullptr == num) {
    return false;
  }
  *num = sysconf(_SC_NPROCESSORS_ONLN);
  return true;
}

bool get_thread_cpu(int64_t *cpu_id) {
  if (nullptr == cpu_id) {
    return false;
  }

  cpu_set_t cpu_mask;
  CPU_ZERO(&cpu_mask);
  int ret = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_mask);
  if (0 != ret) {
    return false;
  }

  for (int i = 0; i < sysconf(_SC_NPROCESSORS_CONF); ++i) {
    if (CPU_ISSET(i, &cpu_mask)) {
      *cpu_id = i;
      return true;
    }
  }

  return false;
}

bool set_thread_cpu(int64_t cpu_id) {
  pthread_t tid = pthread_self();

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_id, &cpu_set);
  int ret = pthread_setaffinity_np(tid, sizeof(cpu_set), &cpu_set);
  if (0 != ret) {
    return false;
  }

  return true;
}

#endif
