// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_OS_SEMAPHORE_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_OS_SEMAPHORE_H_

#include <stdint.h>
#include <semaphore.h>

class Semaphore {
 public:
  Semaphore() = delete;
  Semaphore(const Semaphore&) = delete;

  explicit Semaphore(uint32_t value);
  ~Semaphore();

  void post();
  void wait();
  bool try_wait();

 private:
#ifdef __APPLE__
  sem_t *sem_;
#elif __linux__
  sem_t sem_;
#endif
};

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_OS_SEMAPHORE_H_
