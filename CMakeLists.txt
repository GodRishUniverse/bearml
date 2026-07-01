cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-15)
set(CMAKE_C_COMPILER /usr/bin/gcc-15)
set(CMAKE_CXX_COMPILER /usr/bin/g++-15)

# check_language(CUDA) below runs nvcc in an isolated try_compile that does NOT
# inherit CMAKE_CUDA_HOST_COMPILER. Without this, nvcc defaults to /usr/bin/g++
# (gcc 16 on Fedora 44) and bails with "unsupported GNU version! gcc versions
# later than 15 are not supported", making the probe report NOTFOUND. Setting
# CUDAHOSTCXX in the environment is the documented escape hatch.
set(ENV{CUDAHOSTCXX} "/usr/bin/g++-15")

# CUDA is declared as an OPTIONAL language so the project still configures
# on machines without a CUDA toolkit / GPU (CPU-only builds).
project(SimpleNet CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Host TUs use C++23 so std::bfloat16_t (<stdfloat>) is available for the
# supported-scalar checks. NVCC stays at C++17 (it does not support C++23), so
# .cu files never see std::bfloat16_t and rely on __nv_bfloat16 instead.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)

# User-facing switch. ON by default, but auto-disabled below if no CUDA
# compiler is present. Force OFF with -DSIMPLENET_USE_CUDA=OFF.
option(SIMPLENET_USE_CUDA "Build with CUDA GPU support" ON)

include(CheckLanguage)
check_language(CUDA)

# SIMPLENET_CUDA_ENABLED is the *effective* flag: a plain (non-cache) variable
# recomputed every configure. It is true only when the user hasn't disabled
# CUDA AND a CUDA compiler actually exists. The cached option above is left
# untouched (never FORCE-written) so it can never get stuck.
if(SIMPLENET_USE_CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(SIMPLENET_CUDA_ENABLED TRUE)
    message(STATUS "SimpleNet: CUDA support ENABLED")
  else()
    set(SIMPLENET_CUDA_ENABLED FALSE)
    message(STATUS "SimpleNet: CUDA requested but no CUDA compiler found -> CPU-only build")
  endif()
else()
  set(SIMPLENET_CUDA_ENABLED FALSE)
  message(STATUS "SimpleNet: CUDA support DISABLED by user (CPU-only build)")
endif()

# Add subdirectories
add_subdirectory(simplenet)
add_subdirectory(runchecks)
add_subdirectory(unit_tests)
add_subdirectory(third_party/googletest)
enable_testing()
