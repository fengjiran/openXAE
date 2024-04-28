cmake_minimum_required(VERSION 3.22)
project(openXAE_test)
set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)

file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp)
message(${PROJECT_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${TEST_SRC_FILES})

target_link_libraries(${PROJECT_NAME}
        #        GTest::gtest
        GTest::gtest_main
)