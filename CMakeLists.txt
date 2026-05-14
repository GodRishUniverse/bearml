cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)
set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)

project(SimpleNet CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
else()
  message(STATUS "No CUDA compiler found")
endif()

# Add subdirectories
add_subdirectory(simplenet)
add_subdirectory(runchecks)
add_subdirectory(unit_tests)
add_subdirectory(third_party/googletest)
enable_testing()
