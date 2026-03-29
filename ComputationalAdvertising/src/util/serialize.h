// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_UTIL_SERIALIZE_H_
#define COMPUTATIONALADVERTISING_SRC_UTIL_SERIALIZE_H_

#include <string>
#include "google/protobuf/message.h"

namespace computational_advertising {

std::string pb_to_json(const google::protobuf::Message& pb_msg);
void json_to_pb(const std::string& json_msg, google::protobuf::Message *pb_msg);

}

#endif  // COMPUTATIONALADVERTISING_SRC_UTIL_SERIALIZE_H_
