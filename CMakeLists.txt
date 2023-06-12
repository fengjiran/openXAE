cmake_minimum_required(VERSION 3.22)
project(openXAE)

set(CMAKE_CXX_STANDARD 17)

add_library(openXAE SHARED library.cpp)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "Current platform: Linux ")
    include_directories(/home/richard/project/tvm/include
            /home/richard/project/tvm/3rdparty/dlpack/include
#            /home/richard/project/tvm/3rdparty/dmlc-core/include
            )
    target_link_libraries(${PROJECT_NAME} /home/richard/project/tvm/build/libtvm_runtime.so)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    MESSAGE(STATUS "Current Platform: MacOS")
    include_directories(/Users/richard/Desktop/tvm/include
            /Users/richard/Desktop/tvm/3rdparty/dlpack/include
#            /Users/richard/Desktop/tvm/3rdparty/dmlc-core/include
            )
    target_link_libraries(${PROJECT_NAME} /Users/richard/Desktop/tvm/cmake-build-debug/libtvm_runtime.dylib)
ELSE ()
    MESSAGE(STATUS "Other Platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
