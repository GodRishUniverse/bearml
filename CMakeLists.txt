cmake_minimum_required(VERSION 3.20)
project(SimpleNet CXX)

set(CMAKE_CXX_STANDARD 20)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)




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
