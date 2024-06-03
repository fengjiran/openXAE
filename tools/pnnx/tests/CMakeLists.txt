find_package(GTest REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME}_test ${TEST_SRC_FILES})

target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}_lib
        ${TORCH_LIBRARIES}
        GTest::gtest_main
        glog::glog
)
