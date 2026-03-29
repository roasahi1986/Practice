load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

config_setting(
  name = "use_jemalloc",
  values = {"define": "malloc=jemalloc"},
)

config_setting(
  name = "use_tcmalloc",
  values = {"define": "malloc=tcmalloc"},
)

proto_library(
  name = "perf",
  srcs = ["proto/perf.proto"],
)

cc_proto_library(
  name = "perf_cc",
  deps = [":perf"],
  visibility = ["//visibility:public"],
)

# cc_library(
#   name = "util_os",
#   hdrs = select({
#     "@bazel_tools//src/conditions:darwin_x86_64": glob([
#       "util/os/macos/macros.h",
#       "util/os/macos/thread_cpu.h",
#       "util/os/macos/vpopen.h",
#       "util/os/macos/network.h",
#       "util/os/macos/resource_used.h",
#       "util/os/macos/semaphore.h",
#     ]),
#     "@bazel_tools//src/conditions:darwin": glob([
#       "util/os/macos/macros.h",
#       "util/os/macos/thread_cpu.h",
#       "util/os/macos/vpopen.h",
#       "util/os/macos/network.h",
#       "util/os/macos/resource_used.h",
#       "util/os/macos/semaphore.h",
#     ]),
#     "//conditions:default": glob([
#       "util/os/linux/macros.h",
#       "util/os/linux/thread_cpu.h",
#       "util/os/linux/vpopen.h",
#       "util/os/linux/network.h",
#       "util/os/linux/resource_used.h",
#       "util/os/linux/semaphore.h",
#     ]),
#   }),
#   strip_include_prefix = select({
#     "@bazel_tools//src/conditions:darwin_x86_64": "util/os/macos",
#     "@bazel_tools//src/conditions:darwin": "util/os/macos",
#     "//conditions:default": "util/os/linux",
#   }),
#   include_prefix = "ComputationalAdvertising/src/util/os",
#   visibility = ["//visibility:public"],
# )

cc_library(
  name = "util_os",
  hdrs = [
    "util/os/macros.h",
    "util/os/thread_cpu.h",
    "util/os/vpopen.h",
    "util/os/network.h",
    "util/os/resource_used.h",
    "util/os/semaphore.h",
  ],
  srcs = [
    "util/os/thread_cpu.cpp",
    "util/os/vpopen.cpp",
    "util/os/network.cpp",
    "util/os/resource_used.cpp",
    "util/os/semaphore.cpp",
  ],
  strip_include_prefix = "util/os",
  include_prefix = "ComputationalAdvertising/src/util/os",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "util",
  hdrs = [
    "util/comm.h",
    "util/io.h",
    "util/serialize.h",
    "util/algorithm/search.h",
    "util/functional/timer.h",
    "util/process/process_initiator.h",
    "util/process/process_status.h",
  ],
  srcs = [
    "util/comm.cpp",
    "util/io.cpp",
    "util/serialize.cpp",
    "util/process/process_status.cpp",
  ],
  deps = [
    "@protobuf//:protobuf",
    "@com_google_absl//:absl",
    ":util_os",
  ],
  strip_include_prefix = "util",
  include_prefix = "ComputationalAdvertising/src/util",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "config",
  hdrs = [
    "config/gflags.h",
  ],
  srcs = [
    "config/gflags.cpp",
  ],
  deps = [
    "@com_google_absl//:absl",
  ],
  strip_include_prefix = "config",
  include_prefix = "ComputationalAdvertising/src/config",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "sample",
  hdrs = [
    "engine/sample.h"
  ],
  deps = [
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "engine_base",
  hdrs = [
    "engine/engine.h",
  ],
  deps = [
    ":perf_cc",
    ":sample",
    "@com_google_absl//:absl",
    "@bs_thread_pool//:bs_thread_pool",
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "tf_engine",
  hdrs = [
    "engine/tf_engine.h",
  ],
  srcs = [
    "engine/tf_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    "@com_google_absl//:absl",
    "@tensorflow//:tensorflow",
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "tf_gpu_engine",
  hdrs = [
    "engine/tf_engine.h",
    "engine/tf_gpu_engine.h",
  ],
  srcs = [
    "engine/tf_engine.cpp",
    "engine/tf_gpu_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    "@com_google_absl//:absl",
    "@tensorflow//:tensorflow",
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "tf2_engine",
  hdrs = [
    "engine/tf2_engine.h",
  ],
  srcs = [
    "engine/tf2_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    "@com_google_absl//:absl",
    "@tensorflow//:tensorflow_cc",
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "tf2_gpu_engine",
  hdrs = [
    "engine/tf2_gpu_engine.h",
  ],
  srcs = [
    "engine/tf2_gpu_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    ":tf2_engine",
    "@com_google_absl//:absl",
    "@tensorflow//:tensorflow_cc",
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "onnx_engine",
  hdrs = [
    "engine/onnx_engine.h",
  ],
  srcs = [
    "engine/onnx_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    "@com_google_absl//:absl",
    "@onnxruntime//:onnxruntime",
  ],
  strip_include_prefix = "engine",
  include_prefix = "ComputationalAdvertising/src/engine",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "embedding",
  hdrs = [
    "embedding/embedding.h",
  ],
  strip_include_prefix = "embedding",
  include_prefix = "ComputationalAdvertising/src/embedding",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "population_data",
  hdrs = [
    "population/model_spec.h",
  ],
  srcs = [
    "population/model_spec.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    "@com_google_absl//:absl",
    "@nlohmann_json//:nlohmann_json",
  ],
  strip_include_prefix = "population",
  include_prefix = "ComputationalAdvertising/src/population",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "population",
  hdrs = [
    "population/population.h",
    "population/lifecycle.h",
    "population/roster.h",
  ],
  srcs = [
    "population/population.cpp",
    "population/lifecycle.cpp",
    "population/roster.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":embedding",
    ":engine_base",
    ":tf_engine",
    ":onnx_engine",
    ":population_data",
    "@com_google_absl//:absl",
    "@bs_thread_pool//:bs_thread_pool",
    "@nlohmann_json//:nlohmann_json",
  ],
  strip_include_prefix = "population",
  include_prefix = "ComputationalAdvertising/src/population",
  visibility = ["//visibility:public"],
)

cc_binary(
  name = "perf_tf",
  srcs = [
    "bin/select_engine.h",
    "bin/perf.cpp",
  ],
  deps = [
    ":util",
    ":config",
    ":sample",
    ":tf_engine",
    ":population_data",
    "@com_google_absl//:absl",
  ],
  copts = [
    "-DUSE_TF_ENGINE",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
)

cc_binary(
  name = "perf_tf_gpu",
  srcs = [
    "bin/select_engine.h",
    "bin/perf.cpp",
  ],
  deps = [
    ":util",
    ":config",
    ":sample",
    ":tf_gpu_engine",
    ":population_data",
    "@com_google_absl//:absl",
  ],
  copts = [
    "-DUSE_TF_GPU_ENGINE",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
)

cc_binary(
  name = "perf_onnx",
  srcs = [
    "bin/select_engine.h",
    "bin/perf.cpp",
  ],
  deps = [
    ":util",
    ":config",
    ":sample",
    ":onnx_engine",
    ":population_data",
    "@com_google_absl//:absl",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  copts = [
    "-DUSE_ONNXRUNTIME_ENGINE",
  ],
)

cc_test(
  name = "test_tf_engine",
  srcs = ["unittest/engine/test_tf_engine.cpp"],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    ":tf_engine",
    "@com_google_googletest//:gtest",
    "@com_google_absl//:absl",
  ],
  data = [
    "@//data:model_1",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  timeout = "short",
)

cc_test(
  name = "test_onnx_engine",
  srcs = ["unittest/engine/test_onnx_engine.cpp"],
  deps = [
    ":util",
    ":sample",
    ":engine_base",
    ":onnx_engine",
    "@com_google_googletest//:gtest",
    "@com_google_absl//:absl",
  ],
  data = [
    "@//data:model_1",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  timeout = "short",
)

cc_test(
  name = "test_util",
  srcs = ["unittest/util/test.cpp"],
  deps = [
    "@com_google_googletest//:gtest",
    "@com_google_absl//:absl",
    ":util_os",
    ":util",
  ],
  data = [
    "@//data:files",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  timeout = "short",
)

cc_test(
  name = "bm_flat_hash_map",
  srcs = [
    "benchmark/bm_flat_hash_map.cpp",
  ],
  deps = [
    "@com_github_google_benchmark//:benchmark",
    "@com_google_absl//:absl",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  timeout = "short",
)

cc_test(
  name = "bm_tf_engine",
  srcs = [
    "benchmark/bm_tf_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":tf_engine",
    ":population",
    "@bs_thread_pool//:bs_thread_pool",
    "@com_github_google_benchmark//:benchmark",
    "@com_google_absl//:absl",
  ],
  data = [
    "@//data:model_1",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  timeout = "moderate",
)

cc_test(
  name = "bm_onnx_engine",
  srcs = [
    "benchmark/bm_onnx_engine.cpp",
  ],
  deps = [
    ":util",
    ":sample",
    ":onnx_engine",
    ":population",
    "@bs_thread_pool//:bs_thread_pool",
    "@com_github_google_benchmark//:benchmark",
    "@com_google_absl//:absl",
  ],
  data = [
    "@//data:model_2",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
  timeout = "moderate",
)

cc_binary(
  name = "read_tf_trace",
  srcs = [
    "bin/read_tf_trace.cpp",
  ],
  deps = [
    ":util",
    "@tensorflow//:tensorflow",
    "@com_google_absl//:absl",
  ],
  malloc = select({
    ":use_tcmalloc": "@tcmalloc//:tcmalloc",
    ":use_jemalloc": "@jemalloc//:jemalloc",
    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
  }),
)
