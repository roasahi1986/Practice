// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_OS_VPOPEN_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_OS_VPOPEN_H_

#include <cstdint>
#include <cstdio>

FILE *vpopen(const char *cmd_string, const char *type);
int32_t vpclose(FILE *fp);

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_OS_VPOPEN_H_
