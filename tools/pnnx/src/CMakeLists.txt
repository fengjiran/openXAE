file(GLOB_RECURSE PNNX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${PROJECT_NAME}_exe ${PNNX_SRC} main.cc)

add_library(${PROJECT_NAME} SHARED ${PNNX_SRC})

#set_target_properties(${PROJECT_NAME}_lib PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
