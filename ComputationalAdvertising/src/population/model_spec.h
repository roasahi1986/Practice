// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_POPULATION_MODEL_SPEC_H_
#define COMPUTATIONALADVERTISING_SRC_POPULATION_MODEL_SPEC_H_

#include <stdint.h>
#include <fstream>
#include <vector>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "nlohmann/json.hpp"

namespace computational_advertising {

static const char kFeatureTypeCategorical[]          = "sparse";
static const char kFeatureTypeContinuous[]           = "dense";
static const char kAggregatorMean[]                  = "mean";
static const char kAggregatorMadian[]                = "median";
static const char kAggregatorSum[]                   = "sum";
static const char kAggregatorCount[]                 = "count";
static const char kAggregatorMin[]                   = "min";
static const char kAggregatorMax[]                   = "max";
static const char kAggregatorMode[]                  = "mode";
static const char kAggregatorVariance[]              = "variance";
static const char kAggregatorStandardDeviation[]     = "standard deviation";
static const char kAggregatorQuantile[]              = "quantile";
static const char kAggregatorWeightedMean[]          = "weighted mean";
static const char kInputFeatureTypeFieldName[]       = "type";
static const char kInputFeatureSpecisesFieldName[]   = "slot";
static const char kInputFeatureAggregatorFieldName[] = "combiner";
static const char kInputFeatureDimFieldName[]        = "dim";
static const char kInputFeatureOffsetFieldName[]     = "optimized_offset";
static const char kInputFieldName[]                  = "optimized_inputs";
static const char kInputNameFieldName[]              = "name";
static const char kInputShapeFieldName[]             = "dim";
static const char kInputFeatureFieldName[]           = "input_tensors";
static const char kOutputFieldName[]                 = "outputs";

struct FeatureMeta{
  std::string type;
  std::string aggregator;
  int32_t specises;
  int32_t dim;
  int32_t offset;
};

struct ModelMeta {
  absl::flat_hash_map<std::string, std::vector<int32_t>> output_shapes;
  absl::flat_hash_map<std::string, std::vector<int32_t>> input_shapes;
  absl::flat_hash_map<std::string, std::vector<FeatureMeta>> input_features;
  std::string json_file;
  nlohmann::json conf;

  void load(const std::string& meta_file) noexcept(false);

 private:
  void check_format() noexcept(false);
  void parse_output() noexcept(false);
  void parse_input() noexcept(false);
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_POPULATION_MODEL_SPEC_H_
