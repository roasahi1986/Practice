cc_library(
  name = "gtest",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = ["lib/libgtest.a"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "gmock",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = ["lib/libgmock.a"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "gtest_main",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = ["lib/libgtest_main.a"],
  deps = [
    "gtest",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "gmock_main",
  hdrs = glob([
    "include/**/*.h",
  ]),
  includes = [
    "include",
  ],
  srcs = ["lib/libgmock_main.a"],
  deps = [
    "gmock",
  ],
  visibility = ["//visibility:public"],
)