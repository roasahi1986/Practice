// Copyright (C) 2023 lusyu1986@icloud.com

#include <stdint.h>
#include <fstream>

#include "gflags/gflags.h"
#include "absl/log/log.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include "ComputationalAdvertising/src/util/process/process_initiator.h"
#include "ComputationalAdvertising/src/util/serialize.h"

DEFINE_string(trace_pb_file, "", "The file path of the trace protobuf file");
DEFINE_string(trace_json_file, "", "The file path of the trace json file");

int32_t main(int32_t argc, char **argv) {
    computational_advertising::init(argc, argv);

    try {
        // Read the RunMetadata from the file
        tensorflow::RunMetadata run_metadata;
        std::ifstream input(FLAGS_trace_pb_file, std::ios::binary);
        if (!run_metadata.ParseFromIstream(&input)) {
            LOG(ERROR) << "Failed to read RunMetadata from file: " << FLAGS_trace_pb_file << std::endl;
            return 1;
        }

        // Write the RunMetadata to the file
        std::ofstream output(FLAGS_trace_json_file);
        output << computational_advertising::pb_to_json(run_metadata);
    } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
    } catch (...) {
        LOG(ERROR) << "Unknown exception";
    }

    return 0;
}
