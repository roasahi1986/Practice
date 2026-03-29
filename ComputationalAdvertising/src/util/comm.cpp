// Copyright (C) 2022 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/comm.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <spawn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>  // NOLINT
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "ComputationalAdvertising/src/util/os/vpopen.h"
#include "ComputationalAdvertising/src/util/io.h"

using std::string;
using std::mutex;
using std::lock_guard;

namespace computational_advertising {

static char kShell[] = "sh";
static char kShellFlags[] = "-c";

bool execute(const string& cmd) {
  bool ret = true;
  int32_t exit_code = 0;

  char *argv[] = { kShell, kShellFlags, const_cast<char *>(cmd.c_str()), nullptr };
  pid_t pid;
  exit_code = posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, nullptr);
  if (0 != exit_code) {
    return false;
  }

  while (waitpid(pid, &exit_code, 0) < 0) {
    if (errno != EINTR) {
      return false;
    }
  }

  return ret;
}

bool execute(const string& cmd, string *stdo, bool include_stderr) {
  if (nullptr == stdo) {
    return false;
  }

  bool ret = true;
  int32_t exit_code = 0;

  posix_spawn_file_actions_t action;
  exit_code = posix_spawn_file_actions_init(&action);
  if (0 != exit_code) {
    return false;
  }

  int32_t cout_pipe[2];
  do {
    if (pipe(cout_pipe)) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_addclose(&action, cout_pipe[0]);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_adddup2(&action, cout_pipe[1], 1);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_addclose(&action, cout_pipe[1]);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    if (include_stderr) {
      posix_spawn_file_actions_adddup2(&action, 1, 2);
    }

    char *argv[] = { kShell, kShellFlags, const_cast<char *>(cmd.c_str()), nullptr };
    pid_t pid;
    exit_code = posix_spawn(&pid, "/bin/sh", &action, nullptr, argv, nullptr);
    if (0 != exit_code) {
      ret = false;
      break;
    }

    close(cout_pipe[1]);

    while (waitpid(pid, &exit_code, 0) < 0) {
      if (errno != EINTR) {
        ret = false;
        break;
      }
    }

    if (!read_file(cout_pipe[0], stdo)) {
      ret = false;
      break;
    }
  } while (0);

  close(cout_pipe[0]);
  posix_spawn_file_actions_destroy(&action);

  return ret;
}

bool execute(const string& cmd, string *stdo, string *stde) {
  if (nullptr == stdo || nullptr == stde) {
    return false;
  }

  bool ret = true;
  int32_t exit_code = 0;

  posix_spawn_file_actions_t action;
  exit_code = posix_spawn_file_actions_init(&action);
  if (0 != exit_code) {
    return false;
  }

  int32_t cout_pipe[2];
  int32_t cerr_pipe[2];
  do {
    if (pipe(cout_pipe) || pipe(cerr_pipe)) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_addclose(&action, cout_pipe[0]);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_addclose(&action, cerr_pipe[0]);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_adddup2(&action, cout_pipe[1], 1);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_adddup2(&action, cerr_pipe[1], 2);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_addclose(&action, cout_pipe[1]);
    if (0 != exit_code) {
      ret = false;
      break;
    }
    exit_code = posix_spawn_file_actions_addclose(&action, cerr_pipe[1]);
    if (0 != exit_code) {
      ret = false;
      break;
    }

    char *argv[] = { kShell, kShellFlags, const_cast<char *>(cmd.c_str()), nullptr };
    pid_t pid;
    exit_code = posix_spawn(&pid, "/bin/sh", &action, nullptr, argv, nullptr);
    if (0 != exit_code) {
      ret = false;
      break;
    }

    close(cout_pipe[1]);
    close(cerr_pipe[1]);

    while (waitpid(pid, &exit_code, 0) < 0) {
      if (errno != EINTR) {
        ret = false;
        break;
      }
    }

    if (!read_file(cout_pipe[0], stdo)) {
      ret = false;
      break;
    }
    if (!read_file(cerr_pipe[0], stde)) {
      ret = false;
      break;
    }
  } while (0);

  close(cout_pipe[0]);
  close(cerr_pipe[0]);
  posix_spawn_file_actions_destroy(&action);

  return ret;
}

static mutex g_pipe_mtx;

// popen and pclose are not thread-safe
static FILE *guarded_popen(const char* command, const char* type) {
  lock_guard<mutex> lock(g_pipe_mtx);
  return vpopen(command, type);
}

static int32_t guarded_pclose(FILE* stream) {
  lock_guard<mutex> lock(g_pipe_mtx);
  return vpclose(stream);
}

bool execute_vfork(const string& cmd, string *res, int32_t buffer_size) {
  bool ret = true;
  bool need_stdout = (nullptr != res);

  // vfork and open pipe between parent and child process
  FILE *fp = nullptr;
  fp = guarded_popen(
    absl::StrFormat(/*"set -o pipefail; %s"*/"%s", cmd.c_str()).c_str(),
    need_stdout ? "r" : "w"
  );  // NOLINT
  if (nullptr == fp) {
    LOG(ERROR) << "fail to call guarded_popen(), cmd = '" << cmd << "'";
    return false;
  }

  // read from pipe
  if (need_stdout) {
    char *buffer = nullptr;
    if (buffer_size) {
      buffer = reinterpret_cast<char *>(calloc(buffer_size, sizeof(*buffer)));
      if (nullptr == buffer) {
        LOG(WARNING) << "fail to allocate buffer";
      }
      if (0 != setvbuf(fp, buffer, _IOFBF, buffer_size)) {
        LOG(WARNING) << "fail to set buffer";
      }
    }

    ret = read_file(fp, res);
    if (nullptr != buffer) {
      free(buffer);
    }
  }

  // wait for child finish and close pipe
  if (0 != guarded_pclose(fp)) {
    LOG(ERROR) << "fail to call guarded_pcolse(), cmd = '" << cmd << "'";
    return false;
  }

  return ret;
}

}  // namespace computational_advertising
