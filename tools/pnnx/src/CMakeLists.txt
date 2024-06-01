
set(pnnx_exe_src
        ir.cpp
        storezip.cpp
        main.cc
        utils.cpp
)

set(pnnx_lib_src
        ir.cpp
        storezip.cpp
        utils.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(pnnx ${pnnx_exe_src})

add_library(pnnx_lib SHARED ${pnnx_lib_src})

#set_target_properties()
