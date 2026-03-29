# Production image - builds the service and creates a minimal runtime image
# Uses multi-stage build:
#   Stage 1: Build using base image with all dependencies
#   Stage 2: Copy only the binaries to a minimal runtime image

ARG REGISTRY=ghcr.io
ARG REPO=your-username/computationaladvertising

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM ${REGISTRY}/${REPO}/build-base:latest AS builder

WORKDIR /workspace

# Copy source code
COPY . .

# Setup Bazel configuration
RUN cp bazel/bazel_module MODULE.bazel && \
    sed -i "s|\${HOME}|${HOME}|g" MODULE.bazel

# Copy BUILD files to dependencies
RUN deps="absl benchmark bs_thread_pool gflags glog googletest jemalloc libtensorflow nlohmann_json onnxruntime protobuf zlib" && \
    for dep in $deps; do \
      if [ -d "${HOME}/.local/lib/${dep}" ]; then \
        cp -f bazel/${dep}.WORKSPACE ${HOME}/.local/lib/${dep}/WORKSPACE 2>/dev/null || true; \
        cp -f bazel/${dep}.BUILD ${HOME}/.local/lib/${dep}/BUILD 2>/dev/null || true; \
        cp -f bazel/${dep}.MODULE ${HOME}/.local/lib/${dep}/MODULE.bazel 2>/dev/null || true; \
        sed -i "s|\${HOME}|${HOME}|g" ${HOME}/.local/lib/${dep}/MODULE.bazel 2>/dev/null || true; \
      fi; \
    done

# Build
RUN bazelisk build \
    --jobs=10 \
    --compilation_mode=opt \
    --cxxopt='-std=c++2a' \
    --cxxopt='-Wno-unused-parameter' \
    --cxxopt='-fno-omit-frame-pointer' \
    --cxxopt='-fPIC' \
    --define "malloc=jemalloc" \
    //src:perf_tf \
    //src:perf_onnx

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from builder
COPY --from=builder /workspace/bazel-bin/src/perf_tf /usr/local/bin/
COPY --from=builder /workspace/bazel-bin/src/perf_onnx /usr/local/bin/

# Copy shared libraries that are needed at runtime
COPY --from=builder /root/.local/lib/libtensorflow/lib/*.so* /usr/local/lib/
COPY --from=builder /root/.local/lib/onnxruntime/lib/*.so* /usr/local/lib/
COPY --from=builder /root/.local/lib/jemalloc/lib/*.so* /usr/local/lib/

# Update library cache
RUN ldconfig

# Create models directory
# Note: Models should be mounted at runtime or downloaded separately
# If you want to include models in the image, uncomment and ensure data/models exists:
# COPY --from=builder /workspace/data/models /app/models
RUN mkdir -p /app/models

WORKDIR /app

# Default command
ENTRYPOINT ["/usr/local/bin/perf_tf"]
CMD ["--help"]
