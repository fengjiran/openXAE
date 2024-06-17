# PROJECT_NAME = pnnx

# build torch2pnnx lib
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(torch2pnnx_src
        load_torchscript.cpp
)
#file(GLOB_RECURSE PNNX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


add_library(torch2pnnx OBJECT ${torch2pnnx_src})
target_compile_definitions(torch2pnnx PRIVATE BUILD_TORCH2PNNX)
target_compile_options(torch2pnnx PUBLIC "${TORCH_CXX_FLAGS}")

if (TorchVision_FOUND)
    set_property(SOURCE load_torchscript.cpp APPEND PROPERTY COMPILE_DEFINITIONS PNNX_TORCHVISION)
endif ()

# build pnnx exe
set(pnnx_src
        main.cpp
        Attribute.cpp
        Graph.cpp
        ir.cpp
        storezip.cpp
        utils.cpp
)

add_executable(${PROJECT_NAME} ${pnnx_src})

set_property(SOURCE main.cpp APPEND PROPERTY COMPILE_DEFINITIONS BUILD_TORCH2PNNX)
#target_link_libraries(${PROJECT_NAME} PRIVATE torch2pnnx)

target_link_libraries(${PROJECT_NAME} PRIVATE
        torch2pnnx
        ${TORCH_LIBRARIES}
        pthread
        dl
)

# build pnnx_lib for test
add_library(${PROJECT_NAME}_lib STATIC
#        $<TARGET_OBJECTS:torch2pnnx>
        Attribute.cpp
        Graph.cpp
        ir.cpp
        storezip.cpp
        utils.cpp)

target_link_libraries(${PROJECT_NAME}_lib PRIVATE
        torch2pnnx
        ${TORCH_LIBRARIES}
)
