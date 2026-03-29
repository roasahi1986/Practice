# Copyright (C) 2023 lusyu1986@icloud.com

DEFAULT_CLEAN=false
DEFAULT_STATIC_CODE_CHECK=true
DEFAULT_UNIT_TEST=false
DEFAULT_BENCHMARK_TEST=false
DEFAULT_PERF_DEMO_GRAPH=false
DEFAULT_BUILD_TOOLS=false

function log() {
  echo "$(date +"%Y/%m/%d %H:%M:%S")][$1:$2][INFO] $3"
}

function debug_info() {
  cmd=`sed -n $2p $1`
  echo "$(date +"%Y/%m/%d %H:%M:%S")][$1:$2][DEBUG] execute command \"${cmd}\" ..."
}

function error_info() {
  let begin=$2-10
  let end=$2
  cmd=`sed -n ${begin},${end}p $1`
  echo "$(date +"%Y/%m/%d %H:%M:%S")][$1:$2][ERROR] exit with status $3, cmd:"
  echo "${cmd}"
  exit $3
}

linux_only_repo='\n\nbazel_dep(name = "dnnl")\nlocal_path_override(\n
    module_name = "dnnl",\n
    path = "${HOME}/.local/lib/dnnl",\n)\n
\nbazel_dep(name = "onnxruntime_dnnl")\nlocal_path_override(\n
    module_name = "onnxruntime_dnnl",\n
    path = "${HOME}/.local/lib/onnxruntime_dnnl",\n)\n\nbazel_dep(name = "apache_tvm")\nlocal_path_override(\n
    module_name = "apache_tvm",\n
    path = "${HOME}/.local/lib/apache_tvm",\n)\n
'

linux_only_target='\ncc_library(\n
  name = "onnx_dnnl_engine",\n
  hdrs = [\n
    "engine/onnx_engine.h",\n
    "engine/onnx_dnnl_engine.h",\n
  ],\n
  srcs = [\n
    "engine/onnx_engine.cpp",\n
    "engine/onnx_dnnl_engine.cpp",\n
  ],\n
  deps = [\n
    ":util",\n
    ":sample",\n
    ":engine_base",\n
    "@com_google_absl//:absl",\n
    "@onnxruntime_dnnl//:onnxruntime",\n
  ],\n
  strip_include_prefix = "engine",\n
  include_prefix = "model_server/src/engine",\n
  visibility = ["//visibility:public"],\n)\n\ncc_library(\n
  name = "tvm_engine",\n
  hdrs = [\n
    "engine/tvm_engine.h",\n
  ],\n
  srcs = [\n
    "engine/tvm_engine.cpp",\n
  ],\n
  deps = [\n
    ":util",\n
    ":sample",\n
    ":engine_base",\n
    ":onnx_engine",\n
    "@com_google_absl//:absl",\n
    "@apache_tvm//:tvm",\n
  ],\n
  strip_include_prefix = "engine",\n
  include_prefix = "model_server/src/engine",\n
  visibility = ["//visibility:public"],\n)\n
'

function setup_bazel() {
  cp bazel/bazel_workspace ./WORKSPACE

  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    sed -i "" "s|\${HOME}|${HOME}|g" WORKSPACE
  elif [[ "${uname}" == "Linux" ]]; then
    cp bazel/bazel_rc ./.bazelrc
    sed -i "s|\${HOME}|${HOME}|g" WORKSPACE
  else
    log ${SCRIPT_NAME} ${LINENO} "unknown operating system ${uname}"
    exit 1
  fi
}

function setup_bazel_module() {
  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    cp bazel/bazel_module MODULE.bazel
    sed -i "" "s|\${HOME}|${HOME}|g" MODULE.bazel
  elif [[ "${uname}" == "Linux" ]]; then
    cp bazel/bazel_module MODULE.bazel
    echo -e ${linux_only_repo} >> MODULE.bazel
    sed -i "s|\${HOME}|${HOME}|g" MODULE.bazel
    git checkout src/BUILD
    echo -e ${linux_only_target} >> src/BUILD
  else
    log ${SCRIPT_NAME} ${LINENO} "unknown operating system ${uname}"
    exit 1
  fi

  if [[ -d "${HOME}/.local/lib/absl" ]]; then
    cp -f bazel/absl.WORKSPACE ${HOME}/.local/lib/absl/WORKSPACE
    cp -f bazel/absl.BUILD ${HOME}/.local/lib/absl/BUILD
    cp -f bazel/absl.MODULE ${HOME}/.local/lib/absl/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/absl/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/absl/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/benchmark" ]]; then
    cp -f bazel/benchmark.WORKSPACE ${HOME}/.local/lib/benchmark/WORKSPACE
    cp -f bazel/benchmark.BUILD ${HOME}/.local/lib/benchmark/BUILD
    cp -f bazel/benchmark.MODULE ${HOME}/.local/lib/benchmark/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/benchmark/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/benchmark/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/bs_thread_pool" ]]; then
    cp -f bazel/bs_thread_pool.WORKSPACE ${HOME}/.local/lib/bs_thread_pool/WORKSPACE
    cp -f bazel/bs_thread_pool.BUILD ${HOME}/.local/lib/bs_thread_pool/BUILD
    cp -f bazel/bs_thread_pool.MODULE ${HOME}/.local/lib/bs_thread_pool/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/bs_thread_pool/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/bs_thread_pool/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/gflags" ]]; then
    cp -f bazel/gflags.WORKSPACE ${HOME}/.local/lib/gflags/WORKSPACE
    cp -f bazel/gflags.BUILD ${HOME}/.local/lib/gflags/BUILD
    cp -f bazel/gflags.MODULE ${HOME}/.local/lib/gflags/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/gflags/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/gflags/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/glog" ]]; then
    cp -f bazel/glog.WORKSPACE ${HOME}/.local/lib/glog/WORKSPACE
    cp -f bazel/glog.BUILD ${HOME}/.local/lib/glog/BUILD
    cp -f bazel/glog.MODULE ${HOME}/.local/lib/glog/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/glog/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/glog/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/googletest" ]]; then
    cp -f bazel/googletest.WORKSPACE ${HOME}/.local/lib/googletest/WORKSPACE
    cp -f bazel/googletest.BUILD ${HOME}/.local/lib/googletest/BUILD
    cp -f bazel/googletest.MODULE ${HOME}/.local/lib/googletest/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/googletest/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/googletest/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/jemalloc" ]]; then
    cp -f bazel/jemalloc.WORKSPACE ${HOME}/.local/lib/jemalloc/WORKSPACE
    cp -f bazel/jemalloc.BUILD ${HOME}/.local/lib/jemalloc/BUILD
    cp -f bazel/jemalloc.MODULE ${HOME}/.local/lib/jemalloc/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/jemalloc/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/jemalloc/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/libtensorflow" ]]; then
    cp -f bazel/libtensorflow.WORKSPACE ${HOME}/.local/lib/libtensorflow/WORKSPACE
    cp -f bazel/libtensorflow.BUILD ${HOME}/.local/lib/libtensorflow/BUILD
    cp -f bazel/libtensorflow.MODULE ${HOME}/.local/lib/libtensorflow/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/libtensorflow/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/libtensorflow/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/libtensorflow_gpu" ]]; then
    cp -f bazel/libtensorflow_gpu.WORKSPACE ${HOME}/.local/lib/libtensorflow_gpu/WORKSPACE
    cp -f bazel/libtensorflow_gpu.BUILD ${HOME}/.local/lib/libtensorflow_gpu/BUILD
    cp -f bazel/libtensorflow_gpu.MODULE ${HOME}/.local/lib/libtensorflow_gpu/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/libtensorflow_gpu/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/libtensorflow_gpu/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/nlohmann_json" ]]; then
    cp -f bazel/nlohmann_json.WORKSPACE ${HOME}/.local/lib/nlohmann_json/WORKSPACE
    cp -f bazel/nlohmann_json.BUILD ${HOME}/.local/lib/nlohmann_json/BUILD
    cp -f bazel/nlohmann_json.MODULE ${HOME}/.local/lib/nlohmann_json/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/nlohmann_json/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/nlohmann_json/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/onnxruntime" ]]; then
    cp -f bazel/onnxruntime.WORKSPACE ${HOME}/.local/lib/onnxruntime/WORKSPACE
    cp -f bazel/onnxruntime.BUILD ${HOME}/.local/lib/onnxruntime/BUILD
    cp -f bazel/onnxruntime.MODULE ${HOME}/.local/lib/onnxruntime/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/onnxruntime/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/onnxruntime/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/dnnl" ]]; then
    cp -f bazel/dnnl.WORKSPACE ${HOME}/.local/lib/dnnl/WORKSPACE
    cp -f bazel/dnnl.BUILD ${HOME}/.local/lib/dnnl/BUILD
    cp -f bazel/dnnl.MODULE ${HOME}/.local/lib/dnnl/MODULE.bazel
    sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/dnnl/MODULE.bazel
  fi
  if [[ -d "${HOME}/.local/lib/onnxruntime_dnnl" ]]; then
    cp -f bazel/onnxruntime_dnnl.WORKSPACE ${HOME}/.local/lib/onnxruntime_dnnl/WORKSPACE
    cp -f bazel/onnxruntime_dnnl.BUILD ${HOME}/.local/lib/onnxruntime_dnnl/BUILD
    cp -f bazel/onnxruntime_dnnl.MODULE ${HOME}/.local/lib/onnxruntime_dnnl/MODULE.bazel
    sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/onnxruntime_dnnl/MODULE.bazel
  fi
  if [[ -d "${HOME}/.local/lib/apache_tvm" ]]; then
    cp -f bazel/apache_tvm.WORKSPACE ${HOME}/.local/lib/apache_tvm/WORKSPACE
    cp -f bazel/apache_tvm.BUILD ${HOME}/.local/lib/apache_tvm/BUILD
    cp -f bazel/apache_tvm.MODULE ${HOME}/.local/lib/apache_tvm/MODULE.bazel
    sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/apache_tvm/MODULE.bazel
  fi
  if [[ -d "${HOME}/.local/lib/protobuf" ]]; then
    cp -f bazel/protobuf.WORKSPACE ${HOME}/.local/lib/protobuf/WORKSPACE
    cp -f bazel/protobuf.BUILD ${HOME}/.local/lib/protobuf/BUILD
    cp -f bazel/protobuf.MODULE ${HOME}/.local/lib/protobuf/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/protobuf/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/protobuf/MODULE.bazel
    fi
  fi
  if [[ -d "${HOME}/.local/lib/zlib" ]]; then
    cp -f bazel/zlib.WORKSPACE ${HOME}/.local/lib/zlib/WORKSPACE
    cp -f bazel/zlib.BUILD ${HOME}/.local/lib/zlib/BUILD
    cp -f bazel/zlib.MODULE ${HOME}/.local/lib/zlib/MODULE.bazel
    if [[ "${uname}" == "Darwin" ]]; then
      sed -i "" "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/zlib/MODULE.bazel
    elif [[ "${uname}" == "Linux" ]]; then
      sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/zlib/MODULE.bazel
    fi
  fi
}

function setup() {
  bazel_version=$(cat .bazelversion)

  # Extract major version
  major_version=$(echo "$bazel_version" | cut -d. -f1)

  # Check if major version is smaller than or equal to 6
  if [[ $major_version -le 6 ]]; then
    setup_bazel
  else
    setup_bazel_module
  fi
}

function glob() {
  path=$1
  regex=$2

  for file in ${path}/* ; do
    if [[ -d "${file}" ]]; then
      glob ${file} ${regex}
    elif [[ -f "${file}" ]]; then
      if [[ "${file}" =~ ${regex} ]]; then
        echo ${file}
      fi
    fi
  done
}

function static_code_check() {
  srcs=`glob ./src ".*\.(c|cc|cpp|cxx|c\+\+|C|h|hh|hpp|hxx|inc)$"`
  cpplint.py --verbose=0         \
             --linelength=120    \
             --counting=detailed \
             --repository=..     \
             ${srcs}
  if [[ $? -ne 0 ]]; then
    return 1
  fi
}

function bazel_build() {
  bazelisk build                       \
    --jobs=10                          \
    --compilation_mode opt             \
    --cxxopt='-std=c++2a'              \
    --cxxopt='-Wno-unused-parameter'   \
    --cxxopt='-fno-omit-frame-pointer' \
    --cxxopt='-fPIC'                   \
  "$@"
  if [[ $? -ne 0 ]]; then
    return 1
  fi
}

function bazel_test() {
  bazelisk test                        \
    --compilation_mode opt             \
    --jobs=10                          \
    --test_output=all                  \
    --verbose_failures                 \
    --sandbox_debug                    \
    --test_verbose_timeout_warnings    \
    --dynamic_mode=off                 \
    --spawn_strategy=standalone        \
    --strategy=Genrule=standalone      \
    --cxxopt='-std=c++2a'              \
    --cxxopt='-Wno-unused-parameter'   \
    --cxxopt='-fno-omit-frame-pointer' \
    --cxxopt='-fPIC'                   \
    "$@"
  if [[ $? -ne 0 ]]; then
    return 1
  fi
}

function unit_test() {
  bazel_test //src:test_util --define "malloc=jemalloc" 
  if [[ $? -ne 0 ]]; then
    return 1
  fi
  bazel_test //src:test_tf_engine   --define "malloc=jemalloc"
  if [[ $? -ne 0 ]]; then
    return 1
  fi
  bazel_test //src:test_onnx_engine --define "malloc=jemalloc" 
  if [[ $? -ne 0 ]]; then
    return 1
  fi
}

function benchmark_test() {
  bazel_test //src:bm_flat_hash_map --define "malloc=jemalloc"
  if [[ $? -ne 0 ]]; then
    return 1
  fi

  bazel_test //src:bm_tf_engine --define "malloc=jemalloc" --test_arg="--benchmark_format=console"
  if [[ $? -ne 0 ]]; then
    return 1
  fi

  bazel_test //src:bm_onnx_engine --define "malloc=jemalloc" --test_arg="--benchmark_format=console"
  if [[ $? -ne 0 ]]; then
    return 1
  fi
}

function check() {
  if [[ "${STATIC_CODE_CHECK}" = true || "${DEFAULT_STATIC_CODE_CHECK}" = true ]]; then
    log ${SCRIPT_NAME} ${LINENO} "static analysis is running ..."
    static_code_check
    if [[ $? -ne 0 ]]; then
      log ${SCRIPT_NAME} ${LINENO} "static analysis failed."
      return 1
    fi
  else
    log ${SCRIPT_NAME} ${LINENO} "static analysis is omitted, use STATIC_CODE_CHECK=ture to enable it."
  fi
  
  if [[ "${UNIT_TEST}" = true || "${DEFAULT_UNIT_TEST}" = true ]]; then
    log ${SCRIPT_NAME} ${LINENO} "unit test is running ..."
    unit_test
    if [[ $? -ne 0 ]]; then
      log ${SCRIPT_NAME} ${LINENO} "unit test failed."
      return 1
    fi
  else
    log ${SCRIPT_NAME} ${LINENO} "unit test is omitted, use UNIT_TEST=true to enable it."
  fi
  
  if [[ "${BENCHMARK_TEST}" = true || "${DEFAULT_BENCHMARK_TEST}" = true ]]; then
    log ${SCRIPT_NAME} ${LINENO} "benchmark test is running ..."
    benchmark_test
    if [[ $? -ne 0 ]]; then
      log ${SCRIPT_NAME} ${LINENO} "benchmark test failed."
      return 1
    fi
  else
    log ${SCRIPT_NAME} ${LINENO} "benchmark test is omitted, use BENCHMARK_TEST=true to enable it."
  fi
}

function clean() {
  if [[ "${CLEAN}" = true || "${DEFAULT_CLEAN}" = true ]]; then
    log ${SCRIPT_NAME} ${LINENO} "bazel cleaning ..."
    bazelisk clean --expunge
  else
    log ${SCRIPT_NAME} ${LINENO} "bazel cleaning is omitted, use CLEAN=true to enable it."
  fi
}

function build_tools() {
  if [[ "${BUILD_TOOLS}" = true || "${DEFAULT_BUILD_TOOLS}" = true ]]; then
    tools=("read_tf_trace")
    for tool in ${tools[@]}; do
      bazel_build //src:${tool} --define "malloc=jemalloc"
      if [[ $? -ne 0 ]]; then
        return 1
      fi
    done
  fi
}

function perf_demo_graph() {
  if [[ "${PERF_DEMO_GRAPH}" = true || "${DEFAULT_PERF_DEMO_GRAPH}" = true ]]; then
    log ${SCRIPT_NAME} ${LINENO} "perf demo graph is running ..."
    engine_brands=("tf" "onnx")

    for engine_brand in ${engine_brands[@]}; do
      bazel_build //src:perf_${engine_brand} --define "malloc=jemalloc"
      if [[ $? -ne 0 ]]; then
        return 1
      fi
    done

    uname=`uname`
    if [[ "${uname}" == "Darwin" ]]; then
      for engine_brand in ${engine_brands[@]}; do
        ${SCRIPT_DIR}/bazel-bin/src/perf_${engine_brand} \
          --number_of_test_cases=100                     \
          --number_of_consumers=1                        \
          --engine_opt_level=1                           \
          --engine_jit_level=0
      done
    else
      for engine_brand in ${engine_brands[@]}; do
        numactl --cpunodebind=0 --membind=0                \
          ${SCRIPT_DIR}/bazel-bin/src/perf_${engine_brand} \
          --number_of_test_cases=100                       \
          --number_of_consumers=1                          \
          --engine_opt_level=1                             \
          --engine_jit_level=2
      done
    fi

    if [[ -d "/usr/local/cuda" ]]; then
      bazel_build //src:perf_tf_gpu --define "malloc=jemalloc"
      if [[ $? -ne 0 ]]; then
        return 1
      fi
      ${SCRIPT_DIR}/bazel-bin/src/perf_tf_gpu \
        --number_of_test_cases=100            \
        --number_of_consumers=1               \
        --engine_opt_level=1                  \
        --engine_jit_level=2
    fi
  else
    log ${SCRIPT_NAME} ${LINENO} "perf demo graph is omitted, use PERF_DEMO_GRAPH=true to enable it."
  fi
}

function watch_gpu() {
  watch -n 1 nvidia-smi
  # nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory --format=csv,noheader,nounits -l 1 
}
