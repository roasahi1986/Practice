// Copyright (C) 2021 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/os/resource_used.h"

#ifdef __APPLE__

#include <mach/mach.h>

bool get_process_resource_used(struct ResourceUsed *res) {
  if (nullptr == res) {
    return false;
  }

  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
  kern_return_t ret = task_info(
    mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count);
  if (KERN_SUCCESS != ret) {
    return false;
  }

  res->resident_mb = static_cast<double>(t_info.resident_size) / 1024.0 / 1024.0;
  res->user_time   = static_cast<double>(t_info.user_time.seconds)
                   + static_cast<double>(t_info.user_time.microseconds) / 1000.0;
  res->system_time = static_cast<double>(t_info.system_time.seconds)
                   + static_cast<double>(t_info.system_time.microseconds) / 1000.0;

  return true;
}

bool get_system_resource_used(struct ResourceUsed *res) {
  if (nullptr == res) {
    return false;
  }

  mach_port_t mach_port = mach_host_self();
  vm_size_t page_size;
  kern_return_t ret = host_page_size(mach_port, &page_size);
  if (KERN_SUCCESS != ret) {
    return false;
  }

  mach_msg_type_number_t count;
  vm_statistics64_data_t vm_stats;
  ret = host_statistics64(mach_port, HOST_VM_INFO, (host_info64_t)&vm_stats, &count);
  if (KERN_SUCCESS != ret) {
    return false;
  }

  res->resident_mb = static_cast<double>(
      static_cast<int64_t>(vm_stats.active_count) +
      static_cast<int64_t>(vm_stats.inactive_count) +
      static_cast<int64_t>(vm_stats.wire_count))
    * static_cast<double>(page_size) / 1024.0 / 1024.0;
  res->user_time   = 0.0;
  res->system_time = 0.0;

  return true;
}

#elif defined(__linux__)

#include <time.h>
#include <unistd.h>
#include <string>
#include <ios>
#include <fstream>

using std::string;
using std::ifstream;
using std::ios_base;

bool get_process_resource_used(struct ResourceUsed *res) {
  if (nullptr == res) {
    return false;
  }

  // useless fields
  string pid, comm, state, ppid, pgrp, session, tty_nr,
         tpgid, flags, minflt, cminflt, majflt, cmajflt,
         cutime, cstime, priority, nice, num_threads,
         itrealvalue, starttime, rsslim, startcode, endcode,
         startstack, kstkesp, kstkeip, signal, blocked,
         sigignore, sigcatch, wchan, nswap, cnswap, exit_signal,
         processor, rt_priority, policy, delayacct_blkio_ticks,
         guest_time, cguest_time, start_data, end_data,
         start_brk, arg_start, arg_end, env_start, env_end,
         exit_code;

  // useful fields
  int64_t utime, stime, vsize, rss;

  ifstream stat_stream("/proc/self/stat", ios_base::in);
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss;
  stat_stream.close();

  const double&& page_size_mb = sysconf(_SC_PAGE_SIZE) / 1024.0 / 1024.0;

  // double vm_mb       = vsize / 1024.0 / 1024.0;
  res->resident_mb = rss * page_size_mb;
  res->user_time   = utime / sysconf(_SC_CLK_TCK);
  res->system_time  = stime / sysconf(_SC_CLK_TCK);

  return true;
}

bool get_system_resource_used(struct ResourceUsed *res) {
  if (nullptr == res) {
    return false;
  }

  return true;
}

#endif
