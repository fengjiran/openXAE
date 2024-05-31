cmake_minimum_required(VERSION 3.22)
project(openXAE)

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(PythonInterp REQUIRED)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/tools
)

target_link_libraries(${PROJECT_NAME}
        glog::glog
        blas
        pthread
        ${ARMADILLO_LIBRARY}
)

add_subdirectory(tests)
add_subdirectory(tools/pnnx)
