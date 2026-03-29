cc_library(
  name = "onnxruntime",
  hdrs = glob([
    "include/**",
  ]),
  includes = [
    "include",
  ],
  srcs = select({
    "@bazel_tools//src/conditions:darwin_x86_64": glob([
      # onnxruntime build without --build_shared_lib
      # "lib/**/*.a",
      # onnxruntime build with --build_shared_lib
      "lib/libonnxruntime_providers_shared.so",
      "lib/libonnxruntime.so.1.16.3",
      "lib/libonnxruntime.so",
    ]),
    "@bazel_tools//src/conditions:darwin": glob([
      # onnxruntime build without --build_shared_lib
      # "lib/**/*.a",
      # onnxruntime build with --build_shared_lib
      "lib/libonnxruntime.1.16.3.dylib",
    ]),
    "//conditions:default": glob([
      # onnxruntime build without --build_shared_lib
      # "lib/**/*.a",
      # onnxruntime build with --build_shared_lib
      "lib/libonnxruntime_providers_shared.so",
      "lib/libonnxruntime.so.1.16.3",
      "lib/libonnxruntime.so",
    ]),
  }),
  deps = [
  ],
  linkopts = ["-ldl"], 
  visibility = ["//visibility:public"],
)
