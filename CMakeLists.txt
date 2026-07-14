cmake_minimum_required(VERSION 3.30)

# Host toolchain switch. Must be resolved before project() since that's what
# fixes CMAKE_C/CXX_COMPILER. Pass -DBEARML_COMPILER=clang to opt in.
set(BEARML_COMPILER "gcc" CACHE STRING "Host compiler toolchain: gcc or clang")
set_property(CACHE BEARML_COMPILER PROPERTY STRINGS gcc clang)

if(BEARML_COMPILER STREQUAL "clang")
  find_program(BEARML_C_COMPILER_PATH NAMES clang)
  find_program(BEARML_CXX_COMPILER_PATH NAMES clang++)
  if(NOT BEARML_C_COMPILER_PATH OR NOT BEARML_CXX_COMPILER_PATH)
    message(FATAL_ERROR "BEARML_COMPILER=clang but clang/clang++ not found on PATH")
  endif()
elseif(BEARML_COMPILER STREQUAL "gcc")
  find_program(BEARML_C_COMPILER_PATH NAMES gcc-15 gcc)
  find_program(BEARML_CXX_COMPILER_PATH NAMES g++-15 g++)
  if(NOT BEARML_C_COMPILER_PATH OR NOT BEARML_CXX_COMPILER_PATH)
    message(FATAL_ERROR "BEARML_COMPILER=gcc but gcc/g++ not found on PATH")
  endif()
else()
  message(FATAL_ERROR "BEARML_COMPILER must be 'gcc' or 'clang', got '${BEARML_COMPILER}'")
endif()

set(CMAKE_C_COMPILER ${BEARML_C_COMPILER_PATH})
set(CMAKE_CXX_COMPILER ${BEARML_CXX_COMPILER_PATH})

# nvcc needs a gcc-compatible host compiler; clang host support is version-
# fragile, so CUDA TUs always compile against gcc-15 even in a clang build.
# check_language(CUDA) below runs nvcc in an isolated try_compile that does NOT
# inherit CMAKE_CUDA_HOST_COMPILER. Without this, nvcc defaults to /usr/bin/g++
# (gcc 16 on Fedora 44) and bails with "unsupported GNU version! gcc versions
# later than 15 are not supported", making the probe report NOTFOUND. Setting
# CUDAHOSTCXX in the environment is the documented escape hatch.
find_program(BEARML_CUDA_HOST_COMPILER_PATH NAMES g++-15 g++)
set(CMAKE_CUDA_HOST_COMPILER ${BEARML_CUDA_HOST_COMPILER_PATH})
set(ENV{CUDAHOSTCXX} "${BEARML_CUDA_HOST_COMPILER_PATH}")

# CUDA is declared as an OPTIONAL language so the project still configures
# on machines without a CUDA toolkit / GPU (CPU-only builds).
project(BearML CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Host TUs use C++23 so std::bfloat16_t (<stdfloat>) is available for the
# supported-scalar checks. NVCC stays at C++17 (it does not support C++23), so
# .cu files never see std::bfloat16_t and rely on __nv_bfloat16 instead.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)

# User-facing switch. ON by default, but auto-disabled below if no CUDA
# compiler is present. Force OFF with -DBEARML_USE_CUDA=OFF.
option(BEARML_USE_CUDA "Build with CUDA GPU support" ON)

include(CheckLanguage)
check_language(CUDA)

# BEARML_CUDA_ENABLED is the *effective* flag: a plain (non-cache) variable
# recomputed every configure. It is true only when the user hasn't disabled
# CUDA AND a CUDA compiler actually exists. The cached option above is left
# untouched (never FORCE-written) so it can never get stuck.
if(BEARML_USE_CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(BEARML_CUDA_ENABLED TRUE)
    message(STATUS "BearML: CUDA support ENABLED")
  else()
    set(BEARML_CUDA_ENABLED FALSE)
    message(STATUS "BearML: CUDA requested but no CUDA compiler found -> CPU-only build")
  endif()
else()
  set(BEARML_CUDA_ENABLED FALSE)
  message(STATUS "BearML: CUDA support DISABLED by user (CPU-only build)")
endif()

# Add subdirectories
add_subdirectory(bearml)
add_subdirectory(runchecks)
add_subdirectory(unit_tests)
add_subdirectory(third_party/googletest)
enable_testing()
