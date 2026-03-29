cc_library(
  name = "protobuf",
  hdrs = glob(["include/google/protobuf/**"]),
  srcs = glob(["lib/libprotobuf.a"]),
  includes = ["include"],
  deps = [
    "@com_github_madler_zlib//:zlib",
  ],
  visibility = ["//visibility:public"],
)

filegroup(
  name = "protoc",
  srcs = ["bin/protoc"],
  visibility = ["//visibility:public"],
)

proto_lang_toolchain(
  name = "cc_toolchain",
  command_line = "--cpp_out=$(OUT)",
  runtime = ":protobuf",
  visibility = ["//visibility:public"],
)
