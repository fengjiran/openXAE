file(GLOB_RECURSE torch2pnnx_src ${CMAKE_CURRENT_SOURCE_DIR}/src/torch/*.cpp)
#message(STATUS "Converter_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

if (TorchVision_FOUND)
    set_property(SOURCE src/torch/torch_optimization.cpp APPEND PROPERTY COMPILE_DEFINITIONS PNNX_TORCHVISION)
endif ()

add_library(torch2pnnx OBJECT ${torch2pnnx_src})
target_compile_definitions(torch2pnnx PRIVATE BUILD_TORCH2PNNX)
target_compile_options(torch2pnnx PUBLIC "${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE onnx2pnnx_src ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/*.cpp)
