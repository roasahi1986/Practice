// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_ENGINE_SAMPLE_H_
#define COMPUTATIONALADVERTISING_SRC_ENGINE_SAMPLE_H_

#include <stdint.h>
#include <vector>
#include <string>

namespace computational_advertising {

struct Tensor {
  std::string name;
  int64_t batch_size;
  std::vector<float> data;
};

struct Instance {
  std::vector<Tensor> features;
};

struct Score {
  std::vector<Tensor> targets;
};

struct Sample {
  Instance instance;
  Score score;
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_ENGINE_SAMPLE_H_
