find_package(GTest REQUIRED)
find_package(glog REQUIRED)

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME}_test ${TEST_SRC_FILES})

#message(STATUS "torch_libs = ${TORCH_LIBRARIES}")

target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}_lib
#        ${TORCH_LIBRARY}
#        ${TORCH_LIBRARIES}
#        /home/richard/anaconda3/envs/dl/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so
#        ${c10_LIBRARY}
        GTest::gtest_main
        glog::glog
)
