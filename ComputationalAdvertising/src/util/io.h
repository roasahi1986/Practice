// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_IO_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_IO_H_

#include <cstdio>
#include <string>
#include <string_view>

namespace computational_advertising {

bool read_line(FILE *fp, std::string *line);
bool read_file(FILE *fp, std::string *file);

bool read_line(int32_t fd, std::string *line);
bool read_file(int32_t fd, std::string *file);

bool reverse_file_line(std::string_view input, std::string_view output);

bool curl(const std::string& url, std::string *body);

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_IO_H_
