find_package(GTest REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME}_test ${TEST_SRC_FILES})
target_compile_definitions(${PROJECT_NAME}_test PRIVATE BUILD_TORCH2PNNX)
target_link_libraries(${PROJECT_NAME}_test
        torch2pnnx
        ${PROJECT_NAME}_lib
        GTest::gtest_main
        glog::glog
        ${TORCH_LIBRARIES}
)
