cc_library(
  name = "dnnl",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = [
    "lib/libdnnl.so",
    "lib/libdnnl.so.3",
    "lib/libdnnl.so.3.3",
  ],
  deps = [
  ],
  linkopts = ["-ldl"],
  visibility = ["//visibility:public"],
)
