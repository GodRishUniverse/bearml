cmake_minimum_required(VERSION 3.30)
project(Tests)

set(CMAKE_CXX_STANDARD 20)

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(unit_tests ${TEST_FILES})

# Link against the src library
target_link_libraries(unit_tests
    PRIVATE
        simplenet
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
