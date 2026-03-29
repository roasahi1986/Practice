// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_OS_MACROS_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_OS_MACROS_H_

#ifdef __APPLE__
#define likely(x)       (x)
#define unlikely(x)     (x)
#elif defined(__linux__)
#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#endif  // __GNUC__
#endif  // __APPLE__

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_OS_MACROS_H_
