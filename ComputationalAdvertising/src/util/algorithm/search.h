// Copyright (C) 2021 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_ALGORITHM_SEARCH_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_ALGORITHM_SEARCH_H_

#include <vector>

namespace computational_advertising {

template <typename T, typename Judge>
void binary_search(
  const std::vector<T>& cands,
  const T& target,
  Judger is_valid,
  vector<T>::iterator *result
) {
  (*target) = cands.end();

  int64_t range = static_cast<int64_t>(cand.size());
  range = 1L << (sizeof(range) * 8 - __builtin_clzl(range) - 1);
  for (; range; range >>= 1) {
    if (cands.begin() + range > (*target)) {
      continue;
    } else if (is_valid(target, *((*target) - range))) {
      (*target) -= range;
    }
  }
}

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_ALGORITHM_SEARCH_H_
