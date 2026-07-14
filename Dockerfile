# BearML build image.
#
# Produces a CPU-only build by default. This mirrors CMakeLists.txt's own
# fallback behavior: CUDA support is only enabled when a CUDA compiler is
# present, and this image intentionally doesn't install the CUDA toolkit
# (it's multiple GB and needs a host GPU + nvidia-container-toolkit to be
# useful at all). See the "CUDA support" note below to add it.
FROM fedora:44

# gcc15 / gcc15-c++ provide /usr/bin/gcc-15 and /usr/bin/g++-15, the default
# host compiler resolved by the top-level CMakeLists.txt. `clang` is also
# installed so the image can build with -DBEARML_COMPILER=clang (see the
# build RUN below); it's only ever used for host code, never for .cu files.
RUN dnf install -y \
        gcc15 \
        gcc15-c++ \
        clang \
        cmake \
        make \
        boost-devel \
    && dnf clean all

# --- CUDA support (disabled by default) -------------------------------------
# If the host has an NVIDIA GPU + driver and the NVIDIA Container Toolkit
# (so `docker run --gpus all` works), uncomment the block below to install
# the CUDA toolkit inside the image, then swap which `cmake -S . -B build`
# line further down is active (CPU line commented out, CUDA line uncommented).
#
# RUN dnf install -y https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo \
#     && dnf install -y cuda-toolkit \
#     && dnf clean all
# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV CUDAHOSTCXX="/usr/bin/g++-15"
# -----------------------------------------------------------------------------

WORKDIR /bearml

# Eigen and FFTW are vendored directly under third_party/ and are copied in
# as-is. googletest is the one real git submodule (third_party/googletest) -
# make sure the repo was cloned with `git clone --recursive`, or run
# `git submodule update --init --recursive` before `docker build`, otherwise
# this COPY brings in an empty directory and the build fails.
COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEARML_USE_CUDA=OFF \
    && cmake --build build -j"$(nproc)"
# CUDA build: comment out the CPU-only RUN above and uncomment this one
# instead (after uncommenting the CUDA toolkit install block above). Build
# and run with GPU access: `docker build ... && docker run --gpus all ...`
#
# RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEARML_USE_CUDA=ON \
#     && cmake --build build -j"$(nproc)"
#
# Clang build: comment out the CPU-only RUN above and uncomment this one to
# build host code with clang/clang++ instead of gcc-15 (CUDA .cu files, if
# enabled, still compile via g++-15 regardless of this flag).
#
# RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEARML_USE_CUDA=OFF -DBEARML_COMPILER=clang \
#     && cmake --build build -j"$(nproc)"

# Verification binaries, built above, ready to run:
#   ./build/runchecks/test_runner   - main testing driver
#   ctest --test-dir build          - gtest-based unit tests
CMD ["/bin/bash"]
