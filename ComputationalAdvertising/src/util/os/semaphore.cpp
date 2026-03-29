// Copyright (C) 2021 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/os/semaphore.h"

#include <cstdio>
#include <cstdlib>

#ifdef __APPLE__

#include <semaphore.h>

#include <cerrno>
#include <type_traits>

static const char kSemaphorePath[] = "/semaphore";

template <typename FUNC, typename... ARGS>
static auto ignore_signal_call(
  FUNC func, ARGS &&... args
) -> std::invoke_result_t<FUNC, ARGS...> {
    for (;;) {
        auto err = std::invoke(func, args...);

        if (err < 0 && errno == EINTR) {
            fprintf(stderr, "Signal is caught. Ignored.");
            continue;
        }

        return err;
    }
}

Semaphore::Semaphore(uint32_t value) {
  if ((sem_ = sem_open(kSemaphorePath, O_CREAT, 0644, value)) == SEM_FAILED) {
    fprintf(stderr, "Fail to open semaphore.");
    exit(EXIT_FAILURE);
  }
}

Semaphore::~Semaphore() {
  if (sem_close(sem_) != 0) {
    fprintf(stderr, "Fail to close semaphore.");
    exit(EXIT_FAILURE);
  }
  if (sem_unlink(kSemaphorePath) != 0) {
    fprintf(stderr, "Fail to unlink semaphore.");
    exit(EXIT_FAILURE);
  }
}

void Semaphore::post() {
  if (sem_post(sem_) != 0) {
    fprintf(stderr, "Fail to post semaphore.");
    exit(EXIT_FAILURE);
  }
}
void Semaphore::wait() {
  if (ignore_signal_call(sem_wait, sem_) != 0) {
    fprintf(stderr, "Fail to wait semaphore.");
    exit(EXIT_FAILURE);
  }
}

bool Semaphore::try_wait() {
  int err = ignore_signal_call(sem_trywait, sem_);
  if (0 != err && EAGAIN != errno) {
    fprintf(stderr, "Fail to trywait semaphore.");
    exit(EXIT_FAILURE);
  }
  return 0 == err;
}

// #include <mutex>
// #include <condition_variable>
//
// class Semaphore {
//  public:
//   Semaphore() : count(1) {}
//   explicit Semaphore(int value) : count(value) {}
//   void down() {
//     std::unique_lock<std::mutex> lck(mtk);
//     if (--count < 0) {
//       cv.wait(lck);
//     }
//   }
//   void up() {
//     std::unique_lock<std::mutex> lck(mtk);
//     if (++count <= 0) {
//       cv.notify_one();
//     }
//   }
//  private:
//   int count;
//   std::mutex mtk;
//   std::condition_variable cv;
// };

#elif defined(__linux__)

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <semaphore.h>
#include <type_traits>

template <typename FUNC, typename... ARGS>
static auto ignore_signal_call(
  FUNC func, ARGS &&... args
) -> decltype(func(args...)) {
    for (;;) {
        auto err = func(args...);

        if (err < 0 && errno == EINTR) {
            fprintf(stderr, "Signal is caught. Ignored.");
            continue;
        }

        return err;
    }
}

Semaphore::Semaphore(uint32_t value) {
  if (0 != sem_init(&sem_, 0, value)) {
    fprintf(stderr, "Fail to init semaphore.");
    exit(EXIT_FAILURE);
  }
}

Semaphore::~Semaphore() {
  if (0 != sem_destroy(&sem_)) {
    fprintf(stderr, "Fail to destroy semaphore.");
    exit(EXIT_FAILURE);
  }
}

void Semaphore::post() {
  if (0 != sem_post(&sem_)) {
    fprintf(stderr, "Fail to post semaphore.");
    exit(EXIT_FAILURE);
  }
}

void Semaphore::wait() {
  if (0 != ignore_signal_call(sem_wait, &sem_)) {
    fprintf(stderr, "Fail to wait semaphore.");
    exit(EXIT_FAILURE);
  }
}

bool Semaphore::try_wait() {
  int32_t err = ignore_signal_call(sem_trywait, &sem_);
  if (err != 0 && errno != EAGAIN) {
    fprintf(stderr, "Fail to trywait semaphore.");
    exit(EXIT_FAILURE);
  }
  return 0 == err;
}

// #include <mutex>
// #include <condition_variable>
//
// class Semaphore {
//  public:
//   Semaphore() : count(1) {}
//   explicit Semaphore(int value) : count(value) {}
//   void down() {
//     std::unique_lock<std::mutex> lck(mtk);
//     if (--count < 0) {
//       cv.wait(lck);
//     }
//   }
//   void up() {
//     std::unique_lock<std::mutex> lck(mtk);
//     if (++count <= 0) {
//       cv.notify_one();
//     }
//   }
//  private:
//   int count;
//   std::mutex mtk;
//   std::condition_variable cv;
// };

#endif
