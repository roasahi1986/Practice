// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_OS_RESOURCE_USED_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_OS_RESOURCE_USED_H_

struct ResourceUsed {
  double resident_mb;
  double user_time;
  double system_time;
};

bool get_process_resource_used(struct ResourceUsed *res);
bool get_system_resource_used(struct ResourceUsed *res);

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_OS_RESOURCE_USED_H_
