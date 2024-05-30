include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(pnnx_SRCS
        ir.cpp
        storezip.cpp
        main.cpp)

add_executable(pnnx ${pnnx_SRCS})
