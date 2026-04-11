# simplenet/CMakeLists.txt
cmake_minimum_required(VERSION 3.30)

# Create a library target from all source files in subdirectories
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"  # Add .cu files as well
)

add_library(simplenet ${SRC_FILES})

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

target_link_libraries(simplenet PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    Boost::boost
)

set_target_properties(simplenet PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
