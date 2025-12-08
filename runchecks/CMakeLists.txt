cmake_minimum_required(VERSION 3.20)
project(Tests)

set(CMAKE_CXX_STANDARD 20)

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(test_runner ${TEST_FILES})

# Link against the src library
target_link_libraries(test_runner
    PRIVATE
        simplenet
)
