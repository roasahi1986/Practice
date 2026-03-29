// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/population/model_spec.h"

#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"

namespace computational_advertising {

void ModelMeta::load(const std::string& meta_file) noexcept(false) {
  std::ifstream file(meta_file);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open meta file: " + meta_file);
  }
  auto file_cleanup = absl::MakeCleanup([&file](){ file.close(); });

  conf = nlohmann::json::parse(file);
  check_format();
  parse_output();
  parse_input();
}

void ModelMeta::check_format() noexcept(false) {
  if ((!conf.contains(kInputFieldName)) || (!conf[kInputFieldName].is_array())) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + json_file + "] " + "kInputFieldName format error, " + conf.dump();
    throw std::runtime_error(err_msg);
  }
  if ((!conf.contains(kOutputFieldName)) || (!conf[kOutputFieldName].is_array())) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
      + json_file + "] " + "kOutputFieldName format error, " + conf.dump();
    throw std::runtime_error(err_msg);
  }

  const auto& inputs = conf[kInputFieldName];
  for (const auto& input : inputs) {
    if ((!input.contains(kInputNameFieldName)) || (!input.contains(kInputShapeFieldName))) {
      const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
        + json_file + "] " + "kInputNameFieldName or kInputShapeFieldName format error, " + input.dump();
      throw std::runtime_error(err_msg);
    }

    const auto& shape = input[kInputShapeFieldName];
    if (shape.is_array()) {
      for (const auto& dim : shape) {
        if (!dim.is_number()) {
          const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
            + json_file + "] " + "shape field format error, " + shape.dump();
          throw std::runtime_error(err_msg);
        }
      }
    } else if (shape.is_number()) {
    } else {
      const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
        + json_file + "] " + "shape field format error, " + shape.dump();
      throw std::runtime_error(err_msg);
    }

    if (input.contains(kInputFeatureFieldName)) {
      const auto& features = input[kInputFeatureFieldName];
      if (!features.is_array()) {
        const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
          + json_file + "] " + "kInputFeatureFieldName format error, " + features.dump();
        throw std::runtime_error(err_msg);
      }
      for (const auto& feature : features) {
        if (!feature.contains(kInputFeatureTypeFieldName)) {
          const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
            + json_file + "] " + "kInputFeatureTypeFieldName format error, " + feature.dump();
          throw std::runtime_error(err_msg);
        }
        if (!feature.contains(kInputFeatureAggregatorFieldName)) {
          const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
            + json_file + "] " + "kInputFeatureAggregatorFieldName format error, " + feature.dump();
          throw std::runtime_error(err_msg);
        }
        if ((!feature.contains(kInputFeatureSpecisesFieldName)) ||
            (!feature[kInputFeatureSpecisesFieldName].is_number())) {
          const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
            + json_file + "] " + "kInputFeatureSpecisesFieldName format error, " + feature.dump();
          throw std::runtime_error(err_msg);
        }
        if ((!feature.contains(kInputFeatureDimFieldName)) ||
            (!feature[kInputFeatureDimFieldName].is_number())) {
          const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
            + json_file + "] " + "kInputFeatureDimFieldName format error, " + feature.dump();
          throw std::runtime_error(err_msg);
        }
        if ((!feature.contains(kInputFeatureOffsetFieldName)) ||
            (!feature[kInputFeatureOffsetFieldName].is_number())) {
          const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "]["
            + json_file + "] " + "kInputFeatureOffsetFieldName format error, " + feature.dump();
          throw std::runtime_error(err_msg);
        }
      }
    }
  }
}

void ModelMeta::parse_output() noexcept(false) {
  const auto& outputs = conf[kOutputFieldName];
  for (const auto& output : outputs) {
    output_shapes.insert({output, std::vector<int32_t>()});
  }
}

void ModelMeta::parse_input() noexcept(false) {
  const auto& inputs = conf[kInputFieldName];
  for (const auto& input : inputs) {
    const std::string& name = input[kInputNameFieldName];

    input_shapes[name] = std::vector<int32_t>();
    const auto& shape = input[kInputShapeFieldName];
    if (shape.is_array()) {
      for (const auto& dim : shape) {
        input_shapes[name].push_back(dim.get<int32_t>());
      }
    } else if (shape.is_number()) {
      input_shapes[name].push_back(shape.get<int32_t>());
    }

    if (input.contains(kInputFeatureFieldName)) {
      input_features[name] = std::vector<FeatureMeta>();
      const auto& features = input[kInputFeatureFieldName];
      for (const auto& feature : features) {
        input_features[name].push_back(FeatureMeta{
          .type       = feature[kInputFeatureTypeFieldName].get<std::string>(),
          .aggregator = feature[kInputFeatureAggregatorFieldName].get<std::string>(),
          .specises   = feature[kInputFeatureSpecisesFieldName].get<int32_t>(),
          .dim        = feature[kInputFeatureDimFieldName].get<int32_t>(),
          .offset     = feature[kInputFeatureOffsetFieldName].get<int32_t>(),
        });  // NOLINT
      }
    }
  }
}

}  // namespace computational_advertising
