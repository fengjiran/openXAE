cmake_minimum_required(VERSION 3.22)
project(openXAE)

set(CMAKE_CXX_STANDARD 17)

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "Current platform: Linux ")
    set(TVM_INCLUDE /home/richard/project/tvm/include)
    set(DLPACK_INCLUDE /home/richard/project/tvm/3rdparty/dlpack/include)
    set(DMLC_INCLUDE /home/richard/project/tvm/3rdparty/dmlc-core/include)
    set(TVM_RUNTIME_DLL /home/richard/project/tvm/build/libtvm_runtime.so)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    MESSAGE(STATUS "Current Platform: MacOS")
    set(TVM_INCLUDE /Users/richard/Desktop/tvm/include)
    set(DLPACK_INCLUDE /Users/richard/Desktop/tvm/3rdparty/dlpack/include)
    set(DMLC_INCLUDE /Users/richard/Desktop/tvm/3rdparty/dmlc-core/include)
    set(TVM_RUNTIME_DLL /Users/richard/Desktop/tvm/build/libtvm_runtime.dylib)
ELSE ()
    MESSAGE(STATUS "Other Platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

message(STATUS "TVM include path: " ${TVM_INCLUDE})
message(STATUS "DLPACK include path: " ${DLPACK_INCLUDE})
message(STATUS "DMLC include path: " ${DMLC_INCLUDE})
message(STATUS "TVM runtime DLL path: " ${TVM_RUNTIME_DLL})

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp
                            ${PROJECT_SOURCE_DIR}/src/*.cc)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${TVM_INCLUDE}
        ${DLPACK_INCLUDE}
        ${DMLC_INCLUDE}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${TVM_RUNTIME_DLL}
)
