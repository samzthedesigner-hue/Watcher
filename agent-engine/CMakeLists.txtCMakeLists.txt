cmake_minimum_required(VERSION 3.22)
project(agentllm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED LLAMA_DIR)
    message(FATAL_ERROR "Pass -DLLAMA_DIR=/path/to/llama.cpp")
endif()

set(JNI_CLASS "com/aegis/agent/LlamaBridge" CACHE STRING "JNI class path")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama.cpp-build)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(agentllm SHARED agent_jni.cpp)

target_compile_definitions(agentllm PRIVATE JNI_CLASS="${JNI_CLASS}")

target_include_directories(agentllm PRIVATE
        ${LLAMA_DIR}/include
        ${LLAMA_DIR}/ggml/include)

target_link_libraries(agentllm PRIVATE llama log)

# 16 KB page alignment (needed on targetSdk 35 devices) and smaller binary
target_link_options(agentllm PRIVATE
        -Wl,-z,max-page-size=16384
        -Wl,--gc-sections)
