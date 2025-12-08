cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)
set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)

project(SimpleNet CUDA CXX)

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
add_subdirectory(src)
add_subdirectory(runchecks)
add_subdirectory(unit_tests)
add_subdirectory(third_party/googletest)
enable_testing()


#NOT USED AT THE MOMENT

#cmake_minimum_required(VERSION 3.17)
#project(MyProject)

# Enable CUDA support
#enable_language(CUDA)

# Add CUDA compiler flags
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CUDA_STANDARD 20)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

#add_subdirectory(src)

# Add your CUDA source files
#add_executable(MyExecutable src/test.cu)

# Link against CUDA libraries
#target_link_libraries(MyExecutable ${CUDA_LIBRARIES})
