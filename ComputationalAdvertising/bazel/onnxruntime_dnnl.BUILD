cc_library(
  name = "onnxruntime",
  hdrs = glob([
    "include/**",
  ]),
  includes = [
    "include",
  ],
  deps = [
    "@dnnl//:dnnl"
  ],
  srcs = glob([
    # onnxruntime build without --build_shared_lib
    # "lib/**/*.a",

    # onnxruntime build with --build_shared_lib
    "lib/libonnxruntime_providers_dnnl.so",
    "lib/libonnxruntime_providers_shared.so",
    "lib/libonnxruntime.so.1.16.3",
    "lib/libonnxruntime.so",
  ]),
  linkopts = ["-ldl"],
  visibility = ["//visibility:public"],
)
