file(GLOB_RECURSE PNNX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(torch2pnnx OBJECT load_torchscript.cpp)
target_compile_definitions(torch2pnnx PRIVATE BUILD_TORCH2PNNX)
target_compile_options(torch2pnnx PUBLIC "${TORCH_CXX_FLAGS}")

if(TorchVision_FOUND)
    set_property(SOURCE load_torchscript.cpp APPEND PROPERTY COMPILE_DEFINITIONS PNNX_TORCHVISION)
endif()

add_executable(${PROJECT_NAME} ${PNNX_SRC} main.cc)

add_library(${PROJECT_NAME}_lib SHARED ${PNNX_SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${TORCH_LIBRARIES}
        pthread
        dl
)
