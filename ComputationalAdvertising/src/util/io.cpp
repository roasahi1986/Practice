// Copyright (C) 2021 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/io.h"

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "ComputationalAdvertising/src/util/comm.h"

using std::string;
using std::string_view;

namespace computational_advertising {

static const int32_t kBufferSize = 32768;

bool read_line(FILE *fp, string *line) {
  if (nullptr == fp || nullptr == line) {
    return false;
  }
  char *buffer = nullptr;
  buffer = reinterpret_cast<char *>(calloc(kBufferSize, sizeof(*buffer)));
  if (nullptr == buffer) {
    return false;
  }

  while (fgets(buffer, kBufferSize, fp) != nullptr) {
    int len = strlen(buffer);

    if ('\n' == buffer[len - 1]) {
      buffer[len - 1] = '\0';
      --len;
      *line = *line + buffer;
      break;
    }

    *line = *line + buffer;
  }

  free(buffer);

  return true;
}

bool read_file(FILE *fp, string *file) {
  if (nullptr == fp || nullptr == file) {
    return false;
  }

  char *buffer = nullptr;
  buffer = reinterpret_cast<char *>(calloc(kBufferSize, sizeof(*buffer)));
  if (nullptr == buffer) {
    return false;
  }

  while (fgets(buffer, kBufferSize, fp) != nullptr) {
    *file = *file + buffer;
  }

  free(buffer);

  return true;
}

bool read_line(int32_t fd, string *line) {
  if (nullptr == line) {
    return false;
  }

  char buffer[1];

  ssize_t bytes_read = 0;
  while ((bytes_read = read(fd, buffer, 1)) != 0) {
    if (-1 == bytes_read) {
      return false;
    }

    line->push_back(buffer[0]);
  }

  return true;
}

bool read_file(int32_t fd, string *file) {
  if (nullptr == file) {
    return false;
  }

  char *buffer = nullptr;
  buffer = reinterpret_cast<char *>(calloc(kBufferSize, sizeof(*buffer)));
  if (nullptr == buffer) {
    return false;
  }

  ssize_t bytes_read = 0;
  while ((bytes_read = read(fd, buffer, 1)) != 0) {
    if (-1 == bytes_read) {
      return false;
    }
    file->append(buffer, bytes_read);
  }

  free(buffer);

  return true;
}

bool curl(const string& url, string *body) {
  if (nullptr == body) {
    return false;
  }

  return execute(string("curl ") + url, body);
}


bool reverse_file_line(string_view input, string_view output) {
  bool succ = true;
  FILE *ifp = nullptr;
  FILE *ofp = nullptr;

  do {
    ifp = fopen(std::string(input).c_str(), "r");
    if (nullptr == ifp) {
      LOG(ERROR) << "fail to open file \"" << input << "\"";
      succ = false;
      break;
    }
    ofp = fopen(std::string(output).c_str(), "w");
    if (nullptr == ofp) {
      LOG(ERROR) << "fail to open file \"" << output << "\"";
      succ = false;
      break;
    }

    string stack;
    // read input file from the end
    fseek(ifp, -1L, SEEK_END);
    int64_t num_bytes = ftell(ifp) + 1L;
    for (int64_t i = 0; i < num_bytes; ++i) {
      char ch = fgetc(ifp);
      if (ch == '\n') {
        for (
          auto iter = stack.end() - 1;
          iter >= stack.begin();
          --iter
        ) {
          fputc(*iter, ofp);
        }
        fputc('\n', ofp);
        stack.clear();
      } else {
        stack.push_back(ch);
      }
      fseek(ifp, -2L, SEEK_SET);
    }
  } while (0);

  if (nullptr != ofp) {
    fclose(ofp);
  }
  if (nullptr != ifp) {
    fclose(ifp);
  }

  return succ;
}

}  // namespace computational_advertising
