cc_library(
  name = "jemalloc",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = [
    "lib/libjemalloc_pic.a",
  ],
  deps = [
  ],
  linkopts = ["-ldl"],
  visibility = ["//visibility:public"],
)
