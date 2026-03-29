// Copyright (C) 2023 lusyu1986@icloud.com

#include "ComputationalAdvertising/src/util/serialize.h"

#include <string>

#include "absl/log/log.h"
#include "google/protobuf/message.h"
#include "google/protobuf/util/json_util.h"

namespace computational_advertising {

std::string pb_to_json(const google::protobuf::Message& pb_msg) {
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  options.always_print_primitive_fields = true;
  options.preserve_proto_field_names = true;

  std::string json_msg;
  auto ret = google::protobuf::util::MessageToJsonString(pb_msg, &json_msg, options);
  if (!ret.ok()) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
      + std::string(ret.message());
    throw std::runtime_error(err_msg);
  }
  return json_msg;
}

void json_to_pb(const std::string& json_msg, google::protobuf::Message *pb_msg) {
  if (nullptr == pb_msg) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
      + "pb_msg is nullptr";
    throw std::runtime_error(err_msg);
  }

  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;

  auto ret = google::protobuf::util::JsonStringToMessage(json_msg, pb_msg, options);
  if (!ret.ok()) {
    const std::string& err_msg = "[" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "] "
      + std::string(ret.message());
    throw std::runtime_error(err_msg);
  }

  return;
}

}  // namespace computational_advertising
