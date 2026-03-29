// Copyright (C) 2023 lusyu1986@icloud.com

#include "absl/log/log.h"
#include "gtest/gtest.h"
#include "ComputationalAdvertising/src/util/process/process_initiator.h"
#include "ComputationalAdvertising/src/engine/tf_engine.h"

TEST(TFEngine, LoadSuccess) {
  computational_advertising::EngineConf tf_engine_conf {
    .name = "model_1",
    .version = "1.0.0",
    .graph_file_loc = "data/models/model_1/2/graph.pb",
    .input_nodes = {"dense", "sparse_input_unfolded"},
    .output_nodes = {"predict_node", "p0_click", "p0_atc", "p0_order"},
    .opt_level = 0,
    .jit_level = 0,
    .inter_op_parallelism_threads = 1,
    .intra_op_parallelism_threads = 1
  };
  ASSERT_NO_THROW({
    computational_advertising::TFEngine tf_engine(tf_engine_conf);
  });
}

TEST(TFEngine, LoadFailNonExistentGraph) {
  computational_advertising::EngineConf tf_engine_conf {
    .name = "model_1",
    .version = "1.0.0",
    .graph_file_loc = "data/models/model_1/2/non-existent.pb",
    .input_nodes = {"dense", "sparse_input_unfolded"},
    .output_nodes = {"predict_node", "p0_click", "p0_atc", "p0_order"},
    .opt_level = 0,
    .jit_level = 0,
    .inter_op_parallelism_threads = 1,
    .intra_op_parallelism_threads = 1
  };
  ASSERT_THROW({
    computational_advertising::TFEngine tf_engine(tf_engine_conf);
  }, std::runtime_error);
}

int main(int argc, char **argv) {
  computational_advertising::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
