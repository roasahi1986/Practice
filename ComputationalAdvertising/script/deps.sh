# Copyright (C) 2023 lusyu1986@icloud.com

# CC=/usr/lib/llvm-16/bin/clang
# CXX=/usr/lib/llvm-16/bin/clang++
# LD=/usr/lib/llvm-16/bin/ld.lld
# AR=/usr/lib/llvm-16/bin/llvm-ar
# NM=/usr/lib/llvm-16/bin/llvm-nm
# STRIP=/usr/lib/llvm-16/bin/llvm-strip

DEFAULT_SETUP_DEPS=true
DEFAULT_SETUP_PYTHON=false
DEFAULT_SETUP_OS=false

function get_shell_config() {
  shell_name=$(basename "${SHELL}")
  shell_config=${HOME}/.bashrc
  if [[ ${shell_name} == "zsh" ]]; then
    shell_config=${HOME}/.zshrc
  fi
  echo ${shell_config}
}

function setup_os() {
  if ! [[ ${SETUP_OS} = true || ${DEFULAT_SETUP_OS} = true ]]; then
    echo "setup os skipped, use SETUP_OS=true to enable"
    return
  fi

  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install autoconf	libfido2 m4 bazelisk libidn2 mpdecimal ca-certificates libmpc mpfr cairo libomp openjdk \
      coreutils libpng openjdk@11 fontconfig libpthread-stubs openssl@1.1 freetype libtiff openssl@3 gettext      \
      libunistring pcre2 giflib libx11 pixman git libxau pkg-config glib libxcb protobuf gmp libxdmcp readline    \
      graphite2 libxext snappy harfbuzz libxrender wget icu4c little-cms2 xorgproto isl lrzsz xz jpeg-turbo lz4   \
      zlib libcbor lzlib libevent lzo
  else
    apt-get update
    apt-get install -y apt-transport-https build-essential libcurl4-openssl-dev libcppunit-dev   \
      libunwind-dev libevent-dev libsasl2-dev libzstd-dev libssl-dev libbz2-dev liblz4-dev zlib1g-dev binutils-dev \
      ant g++ gcc autoconf libtool automake pkg-config vim wget curl gnupg tree psmisc net-tools nethogs sysstat   \
      iputils-ping htop iotop iftop oprofile libgoogle-perftools-dev git libsqlite3-dev liblzma-dev                \
      libreadline-dev libbz2-dev
  fi
}

function setup_python() {
  if ! [[ ${SETUP_PYTHON} = true || ${DEFAULT_SETUP_PYTHON} = true ]]; then
    echo "setup python skipped, use SETUP_PYTHON=true to enable"
    return
  fi

  if [ -d ~/.pyenv ]; then
    echo "pyenv already installed"
    return
  fi

  shell_config=$(get_shell_config)

  mkdir -p ${HOME}/.local/bin
  curl https://pyenv.run | bash
  echo '
  export PATH="${HOME}/.local/bin:${HOME}/.pyenv/bin:${PATH}"
  eval "$(pyenv init --path)"
  eval "$(pyenv init -)"' >> ${shell_config}
  source ${shell_config}
  pyenv install 3.11.0
  pyenv global 3.11.0
}

function setup_bazel() {
  if [[ -f ${HOME}/.local/bin/bazelisk ]]; then
    echo "bazelisk already installed"
    return
  fi

  uname=`uname`
  url="https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64"
  if [[ "${uname}" == "Darwin" ]]; then
    url="https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-darwin-arm64"
  fi

  mkdir -p ${HOME}/.local/bin
  curl -fsSL ${url} -o ${HOME}/.local/bin/bazelisk
  chmod +x ${HOME}/.local/bin/bazelisk

  shell_config=$(get_shell_config)
  echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> ${shell_config}
}

function setup_cmake() {
  if [[ -d ${HOME}/.local/cmake ]]; then
    echo "cmake already installed"
    return
  fi

  shell_config=$(get_shell_config)
  pushd ${HOME}/.local/build
  curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.31.4/cmake-3.31.4.tar.gz -o cmake-3.31.4.tar.gz
  tar zxvf cmake-3.31.4.tar.gz
  rm cmake-3.31.4.tar.gz
  pushd cmake-3.31.4
  ./bootstrap --prefix=~/.local/cmake
  make -j10 && make install
  if [[ $? -ne 0 ]]; then
    echo "build cmake failed"
    exit 1
  fi
  echo 'export PATH="${HOME}/.local/cmake/bin:${PATH}"' >> ${shell_config}
  source ${shell_config}
  popd
  popd
}

function setup_cpplint() {
  if [[ -f ${HOME}/.local/bin/cpplint.py ]]; then
    echo "cpplint already installed"
    return
  fi

  mkdir -p ${HOME}/.local/bin
  curl -fsSL https://raw.githubusercontent.com/cpplint/cpplint/master/cpplint.py -o ${HOME}/.local/bin/cpplint.py
  chmod +x ${HOME}/.local/bin/cpplint.py
}

function setup_skylib() {
  if [[ -d ${HOME}/.local/lib/bazel-skylib ]]; then
    echo "bazel-skylib already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf bazel-skylib
  git clone https://github.com/bazelbuild/bazel-skylib.git
  pushd bazel-skylib
  git checkout tags/1.4.2 -b 1.4.2
  popd
  mv bazel-skylib ~/.local/lib
  popd
}

function setup_rules_pkg() {
  if [[ -d ${HOME}/.local/lib/rules_pkg ]]; then
    echo "rules_pkg already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf rules_pkg
  git clone https://github.com/bazelbuild/rules_pkg.git
  pushd rules_pkg
  git checkout tags/0.9.1 -b 0.9.1
  popd
  mv rules_pkg ~/.local/lib
  popd
}

function setup_rules_foreign_cc() {
  if [[ -d ${HOME}/.local/lib/rules_foreign_cc ]]; then
    echo "rules_foreign_cc already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf rules_foreign_cc
  git clone https://github.com/bazelbuild/rules_foreign_cc.git
  pushd rules_foreign_cc
  git checkout tags/0.9.0 -b 0.9.0
  popd
  mv rules_foreign_cc ~/.local/lib
  popd
}

function setup_rules_perl() {
  if [[ -d ${HOME}/.local/lib/rules_perl ]]; then
    echo "rules_perl already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf rules_perl
  git clone https://github.com/bazelbuild/rules_perl.git
  pushd rules_perl
  git checkout tags/0.1.0 -b 0.1.0
  popd
  mv rules_perl ~/.local/lib
  popd
}

function setup_rules_python() {
  if [[ -d ${HOME}/.local/lib/rules_python ]]; then
    echo "rules_python already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf rules_python
  git clone https://github.com/bazelbuild/rules_python.git
  pushd rules_python
  git checkout tags/0.25.0 -b 0.25.0
  popd
  mv rules_python ~/.local/lib
  popd
}

function setup_rules_apple() {
  if [[ -d ${HOME}/.local/lib/rules_apple ]]; then
    echo "rules_apple already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf rules_apple
  git clone https://github.com/bazelbuild/rules_apple.git
  pushd rules_apple
  git checkout tags/0.32.0 -b 0.32.0
  popd
  mv rules_apple ~/.local/lib
  popd
}

function setup_rules_fuzzing() {
  if [[ -d ${HOME}/.local/lib/rules_fuzzing ]]; then
    echo "rules_fuzzing already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf rules_fuzzing
  git clone https://github.com/bazelbuild/rules_fuzzing.git
  pushd rules_fuzzing
  git checkout tags/0.3.2 -b 0.3.2
  popd
  mv rules_fuzzing ~/.local/lib
  popd
}

function setup_gflags() {
  if [[ -d ${HOME}/.local/lib/gflags ]]; then
    echo "gflags already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf gflags
  git clone https://github.com/gflags/gflags.git
  pushd gflags
  git checkout tags/v2.2.2 -b v2.2.2
  mkdir build
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/gflags -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fPIC" -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build gflags failed"
    exit 1
  fi
  cmake --build build --target install
  popd
}

function setup_glog() {
  if [[ -d ${HOME}/.local/lib/glog ]]; then
    echo "glog already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf glog
  git clone https://github.com/google/glog.git
  pushd glog
  git checkout tags/v-1.6.0 -b v0.6.0
  mkdir build
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/glog -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build glog failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  popd
}

function setup_googletest() {
  if [[ -d ${HOME}/.local/lib/googletest ]]; then
    echo "googletest already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf googletest
  git clone https://github.com/google/googletest.git
  pushd googletest
  git checkout tags/v1.13.0 -b v1.13.0
  mkdir build
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/googletest -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build googletest failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  popd
}

function setup_google_benchmark() {
  if [[ -d ${HOME}/.local/lib/benchmark ]]; then
    echo "google benchmark already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf benchmark
  git clone https://github.com/google/benchmark.git
  pushd benchmark
  git checkout tags/v1.8.2 -b v1.8.2
  mkdir build
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/benchmark -DCMAKE_BUILD_TYPE=Release -DGOOGLETEST_PATH=../googletest -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build google benchmark failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  popd
}

function setup_llvm() {
  if [[ -d "/usr/lib/llvm-16" ]]; then
    echo "llvm-16 already installed"
    return
  fi

  pushd ${HOME}/.local/build
  curl -fsSL https://apt.llvm.org/llvm.sh -o llvm.sh
  chmod u+x llvm.sh
  sudo ./llvm.sh 16 all
  popd
}

function setup_cuda() {
  if [[ -d "/usr/local/cuda-12.2" ]]; then
    echo "cuda alread installed"
    return
  fi

  pushd ${HOME}/.local/build
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run -o cuda_12.2.2_535.104.05_linux.run
  sudo sh cuda_12.2.2_535.104.05_linux.run --silent --toolkit
  popd
}

function setup_cudnn() {
  if [[ -d ${HOME}/.local/lib/cudnn ]]; then
    echo "cudnn already installed"
    return
  fi

  pushd ${HOME}/.local/build
  curl -fsSL https://developer.download.nvidia.com/compute/cudnn/secure/8.9.6/local_installers/12.x/cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz -o cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz
  tar xvf cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz
  mv cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/* /usr/local/cuda-12.2/targets/x86_64-linux-gnu/lib/
  mv cudnn-linux-x86_64-8.9.6.50_cuda12-archive/include/* /usr/local/cuda-12.2/targets/x86_64-linux-gnu/include/
  popd
}

function setup_tvm() {
  if [[ -d ${HOME}/.local/lib/apache_tvm ]]; then
    echo "apache_tvm already installed"
    return
  fi

  eval "$(PATH=${HOME}/.local/bin:${HOME}/.local/cmake/bin:${HOME}/.pyenv/bin:$PATH pyenv init --path)" && \
  eval "$(PATH=${HOME}/.local/bin:${HOME}/.local/cmake/bin:${HOME}/.pyenv/bin:$PATH pyenv init -)"

  pushd ${HOME}/.local/build
  rm -rf tvm
  git clone https://github.com/apache/tvm tvm
  pushd tvm
  git checkout tags/v0.15.0 -b v0.15.0
  git submodule init && git submodule update --recursive
  mkdir build

  pushd build
  PATH=${HOME}/.local/bin:${HOME}/.local/cmake/bin:${HOME}/.pyenv/bin:$PATH CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  cmake -DUSE_LLVM=/usr/lib/llvm-16/bin/llvm-config -DUSE_CUDA=/usr/local/cuda-11.8 -DUSE_CUDNN=ON           \
    -DUSE_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc        \
    -DCMAKE_INSTALL_PREFIX=~/.local/lib/apache_tvm ..
  make -j32
  make install
  popd
  cp -r 3rdparty/dmlc-core/include/* ~/.local/lib/apache_tvm/include
  cp -r 3rdparty/dlpack/include/* ~/.local/lib/apache_tvm/include

  pushd python
  python3 setup.py install --user
  popd

  popd
  popd
}

function setup_tensorrt() {
  if [[ -d "${HOME}/.local/lib/TensorRT" ]]; then
    echo "tensorrt already installed"
    return
  fi

  pushd ${HOME}/.local/build
  curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.2.tar.gz -o TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.2.tar.gz
  tar zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.2.tar.gz
  mv TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/* /usr/local/cuda-12.2/targets/x86_64-linux-gnu/lib/
  mv TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/stubs/* /usr/local/cuda-12.2/targets/x86_64-linux-gnu/lib/stubs/
  mv TensorRT-8.6.1.6/targets/x86_64-linux-gnu/include/* /usr/local/cuda-12.2/targets/x86_64-linux-gnu/include/ 
  popd
}

function setup_tensorflow() {
  if [[ -d ${HOME}/.local/lib/libtensorflow ]]; then
    echo "tensorflow already installed"
    return
  fi
  # pip install "tensorflow[and-cuda]==2.15"

  if [[ -d "/usr/local/cuda" ]]; then
    export CUDA_TOOLKIT_PATH="/usr/local/cuda-12.2"
    export CUDNN_INSTALL_PATH="/usr/local/cuda-12.2/targets/x86_64-linux-gnu"
    export TF_NEED_CUDA=1
    export TF_NEED_CLANG=1
  fi

  pushd ${HOME}/.local/build
  rm -rf tensorflow
  git clone https://github.com/tensorflow/tensorflow.git
  pushd tensorflow
  git checkout tags/v2.15.0 -b v2.15.0
  yes '' | ./configure
  bazelisk clean --expunge
  uname=`uname`

  compile_flags="--jobs=60 --compilation_mode=opt --config=monolithic --config=nogcp --config=nonccl"
  compile_flags="${compile_flags} --copt=-fpermissive --copt=-Wno-error"
  if [[ "${uname}" == "Darwin" ]]; then
    compile_flags="${compile_flags} --config=macos --config=release_macos_arm64 --config=mkl_aarch64 --spawn_strategy=sandboxed"
  else
    if [[ -d "/usr/local/cuda" ]]; then
      compile_flags="${compile_flags} --config=cuda_clang"
      compile_flags="${compile_flags} --action_env=TF_CUDNN_VERSION=8"
      compile_flags="${compile_flags} --action_env=TF_CUDA_VERSION=11.8"
      compile_flags="${compile_flags} --action_env=CLANG_CUDA_COMPILER_PATH=/usr/lib/llvm-16/bin/clang"
      compile_flags="${compile_flags} --linkopt=-L${CUDA_TOOLKIT_PATH}/lib64"
      compile_flags="${compile_flags} --linkopt=-L${CUDA_TOOLKIT_PATH}/extras/CUPTI/lib64"
      compile_flags="${compile_flags} --linkopt=-lcusparse"
    fi
    compile_flags="${compile_flags} --config=release_linux_base --config=mkl"
  fi
  bazelisk build ${compile_flags} tensorflow/tools/lib_package:libtensorflow //tensorflow:install_headers # //tensorflow:libtensorflow_cc.so
  if [[ $? -ne 0 ]]; then
    echo "build tensorflow failed"
    exit 1
  fi
  mkdir -p ~/.local/lib/libtensorflow
  mkdir -p ~/.local/lib/libtensorflow/libcc
  tar zxvf bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz -C ~/.local/lib/libtensorflow
  cp bazel-bin/tensorflow/libtensorflow_cc.so.2 ~/.local/lib/libtensorflow/libcc
  cp bazel-bin/tensorflow/libtensorflow_framework.so.2.15.0 ~/.local/lib/libtensorflow/libcc/libtensorflow_framework.so.2
  cp -r bazel-bin/tensorflow/core/protobuf ~/.local/lib/libtensorflow/include/tensorflow/core
  cp -r bazel-bin/tensorflow/core/framework ~/.local/lib/libtensorflow/include/tensorflow/core
  # cp -r bazel-bin/external/local_tsl/tsl ~/.local/lib/libtensorflow/include
  find tensorflow/cc tensorflow/core -name \*.h -exec cp --parents \{\} ~/.local/lib/libtensorflow/include \;
  pushd third_party/xla/third_party/tsl
  find tsl -name \*.h -exec cp --parents \{\} ~/.local/lib/libtensorflow/include \;
  popd
  cp -r bazel-bin/external/local_tsl/tsl/protobuf ~/.local/lib/libtensorflow/include/tsl
  cp -r bazel-bin/tensorflow/include/_virtual_includes/float8/ml_dtypes ~/.local/lib/libtensorflow/include
  cp -r bazel-bin/tensorflow/include/_virtual_includes/int4/ml_dtypes ~/.local/lib/libtensorflow/include
  cp -r bazel-bin/tensorflow/include/Eigen ~/.local/lib/libtensorflow/include
  cp -r bazel-bin/tensorflow/include/unsupported ~/.local/lib/libtensorflow/include

  popd
  popd
}

function setup_zlib() {
  if [[ -d ${HOME}/.local/lib/zlib ]]; then
    echo "zlib already installed"
    return
  fi

  pushd ${HOME}/.local/build
  cp -r tensorflow/bazel-tensorflow/external/zlib ./zlib
  pushd zlib
  ./configure --prefix=~/.local/lib/zlib
  make -j10 && make install
  if [[ $? -ne 0 ]]; then
    echo "build zlib failed"
    exit 1
  fi
  popd
  popd
}

function setup_protobuf() {
  if [[ -d ${HOME}/.local/lib/protobuf-src ]]; then
    echo "protobuf already installed"
    return
  fi

  pushd ${HOME}/.local/build
  git clone --depth 1 --branch v33.2 https://github.com/protocolbuffers/protobuf.git
  pushd protobuf
  git submodule update --init --recursive
  cmake -DCMAKE_INSTALL_PREFIX=${HOME}/.local/lib/protobuf \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_ABSL_PROVIDER=package \
        -DCMAKE_PREFIX_PATH=${HOME}/.local/lib/absl \
        -DCMAKE_BUILD_TYPE=Release \
        -S . -B build
  cmake --build build -j$(nproc)
  if [[ $? -ne 0 ]]; then
    echo "build protobuf failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  # Keep source for Bazel rules
  mv protobuf ${HOME}/.local/lib/protobuf-src
  popd
}

function setup_abseil() {
  if [[ -d ${HOME}/.local/lib/absl ]]; then
    echo "absl already installed"
    return
  fi

  pushd ${HOME}/.local/build
  cp -r tensorflow/bazel-tensorflow/external/com_google_absl ./com_google_absl
  pushd com_google_absl
  mkdir build
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/absl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build absl failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  popd
}

function setup_onnx() {
  if [[ -d ${HOME}/.local/lib/onnxruntime ]]; then
    echo "onnxruntime already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf onnxruntime
  git clone https://github.com/microsoft/onnxruntime.git # or (https://github.com/intel/onnxruntime.git)
  pushd onnxruntime
  # git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
  git checkout tags/v1.23.2 -b v1.23.2

  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    ./build.sh --config Release --parallel --build_shared_lib --compile_no_warning_as_error --skip_submodule_sync \
      --cmake_extra_defines CMAKE_INSTALL_PREFIX=~/.local/lib/onnxruntime                                         \
      --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 &&                                                      \
    pushd build/MacOS/Release && make install && popd
  else
    ./build.sh --config Release --parallel --build_shared_lib --compile_no_warning_as_error \
      --cmake_extra_defines CMAKE_INSTALL_PREFIX=~/.local/lib/onnxruntime                   \
      --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=x86_64 &&                               \
    pushd build/Linux/Release && make install && popd
  fi
  if [[ $? -ne 0 ]]; then
    echo "build onnxruntime failed"
    exit 1
  fi
  popd
  popd
}

function setup_dnnl() {
  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    return
  fi
  if [[ -d ${HOME}/.local/lib/dnnl ]]; then
    echo "dnnl already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf mkl-dnn
  git clone https://github.com/intel/mkl-dnn.git
  pushd mkl-dnn
  # git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
  git checkout tags/v3.3.3 -b v3.3.3
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/dnnl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fPIC" -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build dnnl failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  popd
}

function setup_onnx_dnnl() {
  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    return
  fi
  if [[ -d ${HOME}/.local/lib/onnxruntime_dnnl ]]; then
    echo "onnxruntime_dnnl already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf onnxruntime
  git clone https://github.com/microsoft/onnxruntime.git # or (https://github.com/intel/onnxruntime.git)
  pushd onnxruntime
  # git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
  git checkout tags/v1.23.2 -b v1.23.2
  ./build.sh --config Release --parallel --build_shared_lib --compile_no_warning_as_error --use_dnnl \
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=~/.local/lib/onnxruntime_dnnl                         \
    --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=x86_64
  if [[ $? -ne 0 ]]; then
    echo "build onnxruntime_dnnl failed"
    exit 1
  fi
  pushd build/Linux/Release && make install && popd
  popd
  popd
}

function setup_onnx_openvino() {
  uname=`uname`
  if [[ "${uname}" == "Darwin" ]]; then
    return
  fi
  if [[ -d ${HOME}/.local/lib/onnxruntime_openvino ]]; then
    echo "onnxruntime_openvino already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf onnxruntime
  git clone https://github.com/microsoft/onnxruntime.git # or (https://github.com/intel/onnxruntime.git)
  # git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
  pushd onnxruntime
  git checkout tags/v1.23.2 -b v1.23.2
  ./build.sh --config Release --parallel --build_shared_lib --use_openvino CPU_FP32                \
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=~/.local/lib/onnxruntime_openvino              \
    --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds" \
    --cmake_extra_defines CMAKE_C_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds"
  if [[ $? -ne 0 ]]; then
    echo "build onnxruntime_openvino failed"
    exit 1
  fi
  pushd build/Linux/Release && make install && popd
  popd
  popd
}

function setup_nlohmann_json() {
  if [[ -d ${HOME}/.local/lib/nlohmann_json ]]; then
    echo "nlohmann_json already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf json
  git clone https://github.com/nlohmann/json.git
  pushd json
  git checkout tags/v3.11.2 -b v3.11.2
  cmake -DCMAKE_INSTALL_PREFIX=~/.local/lib/nlohmann_json -DCMAKE_BUILD_TYPE=Release -S . -B build
  cmake --build build -j10
  if [[ $? -ne 0 ]]; then
    echo "build nlohmann_json failed"
    exit 1
  fi
  cmake --build build --target install
  popd
  popd
}

function setup_bshoshany_thread_pool() {
  if [[ -d ${HOME}/.local/lib/bs_thread_pool ]]; then
    echo "bs_thread_pool already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf thread-pool
  git clone https://github.com/bshoshany/thread-pool.git
  pushd thread-pool
  git checkout tags/v3.5.0 -b v3.5.0
  mkdir -p ~/.local/lib/bs_thread_pool/include/BShoshany
  mv include/* ~/.local/lib/bs_thread_pool/include/BShoshany
  popd
  popd
}

function setup_jemalloc() {
  if [[ -d ${HOME}/.local/lib/jemalloc ]]; then
    echo "jemalloc already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf jemalloc
  git clone https://github.com/jemalloc/jemalloc.git
  pushd jemalloc
  git checkout tags/5.3.0 -b 5.3.0
  ./autogen.sh
  ./configure --prefix=${HOME}/.local/lib/jemalloc
  make -j10 && make install
  if [[ $? -ne 0 ]]; then
    echo "build jemalloc failed"
    exit 1
  fi
  popd
  popd
}

function setup_tcmalloc() {
  if [[ -d ${HOME}/.local/lib/tcmalloc ]]; then
    echo "tcmalloc already installed"
    return
  fi

  pushd ${HOME}/.local/build
  rm -rf tcmalloc
  git clone https://github.com/google/tcmalloc.git
  mv tcmalloc ~/.local/lib/tcmalloc
  popd
}

function setup_vim_copilot() {
  # https://docs.github.com/en/copilot/using-github-copilot/getting-started-with-github-copilot?tool=vimneovim#prerequisites-3
  brew install node.js
  git clone https://github.com/github/copilot.vim.git ${HOME}/vimfiles/pack/github/start/copilot.vim
  echo "set modifiable" >> ${HOME}/.vimrc
}

function setup_deps() {
  if ! [[ ${SETUP_DEPS} = true || ${DEFAULT_SETUP_DEPS} = true ]]; then
    echo "setup deps skipped, use SETUP_DEPS=true to enable"
    return
  fi

  mkdir -p ${HOME}/.local/build
  setup_os
  setup_python
  setup_bazel
  setup_cmake
  setup_cpplint
  setup_gflags
  setup_glog
  setup_googletest
  setup_google_benchmark
  setup_tensorflow
  setup_zlib
  setup_protobuf
  setup_abseil
  setup_nlohmann_json
  setup_bshoshany_thread_pool
  setup_jemalloc
  setup_tcmalloc
  setup_onnx
  setup_onnx_dnnl
  setup_onnx_openvino
  setup_skylib
  setup_rules_pkg
  setup_rules_foreign_cc
  setup_rules_perl
  setup_rules_python
  setup_rules_apple
  setup_rules_fuzzing

  find /tmp -user ${USER} -type d -delete
}
