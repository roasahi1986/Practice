cc_library(
  name = "gflags_nothreads",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = ["lib/libgflags_nothreads.a"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "gflags",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = ["lib/libgflags.a"],
  linkopts = [
    "-lpthread",
  ],
  visibility = ["//visibility:public"],
)