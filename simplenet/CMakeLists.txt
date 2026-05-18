# simplenet/CMakeLists.txt
cmake_minimum_required(VERSION 3.30)

# Create a library target from all source files in subdirectories.
# .cu files are only compiled into the library when CUDA is enabled.
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(SIMPLENET_CUDA_ENABLED)
    file(GLOB_RECURSE CUDA_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
    list(APPEND SRC_FILES ${CUDA_SRC_FILES})
endif()

add_library(simplenet ${SRC_FILES})

# Source code gates CUDA paths on this macro: #if defined(SIMPLENET_USE_CUDA)
# (kept identical to the CMake option name so the C++ guards and the build
#  switch never drift apart again).
if(SIMPLENET_CUDA_ENABLED)
    target_compile_definitions(simplenet PUBLIC SIMPLENET_USE_CUDA)
endif()

target_include_directories(simplenet
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # <- simplenet/
)

# Add include directories:
# - src/ itself (so you can do #include "simplenet/tensor/Tensor.h")
# - eigen (so you can do #include <Eigen/Dense>)
target_include_directories(simplenet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/third_party/eigen-3.4.0
)


# if(POLICY CMP0167)
#   cmake_policy(SET CMP0167 NEW)
# endif()


find_package(Boost REQUIRED)

target_link_libraries(simplenet PUBLIC Boost::boost)

# Only link the CUDA runtime/driver when CUDA support is enabled.
if(SIMPLENET_CUDA_ENABLED)
    target_link_libraries(simplenet PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    )
    set_target_properties(simplenet PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()
