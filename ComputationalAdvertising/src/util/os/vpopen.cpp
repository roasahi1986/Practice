// Copyright (C) 2021 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/os/vpopen.h"

#include <cstdio>
#include <cstdlib>

#ifdef __APPLE__

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>

static const int64_t kOpenMaxGuess = 1024;
static const char    kShell[] = "/bin/sh";

static pid_t *g_child_pid = nullptr;  /* ptr to array allocated at run-time */
static int32_t g_max_fd = 0;          /* from our open_max(), {Prog openmax} */
static int64_t g_open_max = 0;

/*
 * If OPEN_MAX is indeterminate, we're not
 * guaranteed that this is adequate.
 */
static int64_t open_max()  {
  if (g_open_max == 0) {
    errno = 0;
    if ((g_open_max = sysconf(_SC_OPEN_MAX)) < 0) {
      if (errno == 0) {
        g_open_max = kOpenMaxGuess;
      } else {
        fprintf(stderr, "sysconf error for _SC_OPEN_MAX");
      }
    }
  }

  return g_open_max;
}

FILE *vpopen(const char *cmd_string, const char *type) {
  if ((nullptr == cmd_string) || (nullptr == type)) {
    errno = EINVAL;
    return nullptr;
  }

  /* only allow "r" or "w" */
  if ((type[0] != 'r' && type[0] != 'w') || type[1] != 0) {
    errno = EINVAL;     /* required by POSIX.2 */
    return nullptr;
  }

  if (nullptr == g_child_pid) {
    /* allocate zeroed out array for child pids */
    g_max_fd = open_max();
    g_child_pid = reinterpret_cast<pid_t *>(calloc(g_max_fd, sizeof(pid_t)));
    if (nullptr == g_child_pid) {
      return nullptr;
    }
  }

  int32_t pfd[2];
  if (0 != pipe(pfd)) {
    return nullptr;   /* errno set by pipe() */
  }

  pid_t pid = vfork();
  if (pid < 0) {
    return nullptr;   /* errno set by fork() */
  }

  if (pid == 0) {  /* child */
    if (*type == 'r') {
      close(pfd[0]);
      if (pfd[1] != STDOUT_FILENO) {
        dup2(pfd[1], STDOUT_FILENO);
        close(pfd[1]);
      }
    } else {
      close(pfd[1]);
      if (pfd[0] != STDIN_FILENO) {
        dup2(pfd[0], STDIN_FILENO);
        close(pfd[0]);
      }
    }

    /*
     * close all descriptors in g_child_pid[],
     * to avoid other tasks use stdin or stdout simultaneously
     */
    for (int32_t i = 0; i < g_max_fd; ++i) {
      if (g_child_pid[i] > 0) {
        close(i);
      }
    }

    execl(kShell, "sh", "-c", cmd_string, nullptr);
    _exit(127);
  }

  /* parent */
  FILE *fp = nullptr;
  if (*type == 'r') {
    close(pfd[1]);
    if (nullptr == (fp = fdopen(pfd[0], type))) {
      return nullptr;
    }
  } else {
    close(pfd[0]);
    if (nullptr == (fp = fdopen(pfd[1], type))) {
      return nullptr;
    }
  }
  g_child_pid[fileno(fp)] = pid; /* remember child pid for this fd */

  return fp;
}

int32_t vpclose(FILE *fp) {
  if (nullptr == fp || nullptr == g_child_pid) { /* popen() has never been called */
    return -1;
  }

  int32_t fd = fileno(fp);
  if (fd >= open_max()) {
    return -1;
  }

  pid_t pid = g_child_pid[fd];
  if (0 == pid) {
    return -1; /* fp wasn't opened by popen() */
  }

  g_child_pid[fd] = 0;
  if (fclose(fp) == EOF) {
    return -1;
  }

  int32_t stat = 0;
  while (waitpid(pid, &stat, 0) < 0) {
    if (errno != EINTR) {
      return -1; /* error other than EINTR from waitpid() */
    }
  }

  return stat; /* return child's termination status */
}

#elif defined(__linux__)

#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

static const int64_t kOpenMaxGuess = 1024;
static const char    kShell[] = "/bin/sh";

static pid_t *g_child_pid = nullptr;  /* ptr to array allocated at run-time */
static int32_t g_max_fd = 0;          /* from our open_max(), {Prog openmax} */
static int64_t g_open_max = 0;

/*
 * If OPEN_MAX is indeterminate, we're not
 * guaranteed that this is adequate.
 */
static int64_t open_max()  {
  if (g_open_max == 0) {
    errno = 0;
    if ((g_open_max = sysconf(_SC_OPEN_MAX)) < 0) {
      if (errno == 0) {
        g_open_max = kOpenMaxGuess;
      } else {
        fprintf(stderr, "sysconf error for _SC_OPEN_MAX");
      }
    }
  }

  return g_open_max;
}

FILE *vpopen(const char *cmd_string, const char *type) {
  if ((nullptr == cmd_string) || (nullptr == type)) {
    errno = EINVAL;
    return nullptr;
  }

  /* only allow "r" or "w" */
  if ((type[0] != 'r' && type[0] != 'w') || type[1] != 0) {
    errno = EINVAL;     /* required by POSIX.2 */
    return nullptr;
  }

  if (nullptr == g_child_pid) {
    /* allocate zeroed out array for child pids */
    g_max_fd = open_max();
    g_child_pid = reinterpret_cast<pid_t *>(calloc(g_max_fd, sizeof(pid_t)));
    if (nullptr == g_child_pid) {
      return nullptr;
    }
  }

  int32_t pfd[2];
  if (0 != pipe(pfd)) {
    return nullptr;   /* errno set by pipe() */
  }

  pid_t pid = vfork();
  if (pid < 0) {
    return nullptr;   /* errno set by fork() */
  }

  if (pid == 0) {  /* child */
    if (*type == 'r') {
      close(pfd[0]);
      if (pfd[1] != STDOUT_FILENO) {
        dup2(pfd[1], STDOUT_FILENO);
        close(pfd[1]);
      }
    } else {
      close(pfd[1]);
      if (pfd[0] != STDIN_FILENO) {
        dup2(pfd[0], STDIN_FILENO);
        close(pfd[0]);
      }
    }

    /*
     * close all descriptors in g_child_pid[],
     * to avoid other tasks use stdin or stdout simultaneously
     */
    for (int32_t i = 0; i < g_max_fd; ++i) {
      if (g_child_pid[i] > 0) {
        close(i);
      }
    }

    execl(kShell, "sh", "-c", cmd_string, nullptr);
    _exit(127);
  }

  /* parent */
  FILE *fp = nullptr;
  if (*type == 'r') {
    close(pfd[1]);
    if (nullptr == (fp = fdopen(pfd[0], type))) {
      return nullptr;
    }
  } else {
    close(pfd[0]);
    if (nullptr == (fp = fdopen(pfd[1], type))) {
      return nullptr;
    }
  }
  g_child_pid[fileno(fp)] = pid; /* remember child pid for this fd */

  return fp;
}

int32_t vpclose(FILE *fp) {
  if (nullptr == fp || nullptr == g_child_pid) { /* popen() has never been called */
    return -1;
  }

  int32_t fd = fileno(fp);
  if (fd >= open_max()) {
    return -1;
  }

  pid_t pid = g_child_pid[fd];
  if (0 == pid) {
    return -1; /* fp wasn't opened by popen() */
  }

  g_child_pid[fd] = 0;
  if (fclose(fp) == EOF) {
    return -1;
  }

  int32_t stat = 0;
  while (waitpid(pid, &stat, 0) < 0) {
    if (errno != EINTR) {
      return -1; /* error other than EINTR from waitpid() */
    }
  }

  return stat; /* return child's termination status */
}

#endif
