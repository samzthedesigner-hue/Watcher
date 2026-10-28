cmake_minimum_required(VERSION 3.22)
project(watcher_jni C CXX)

# ---- Tunables (override with -DWA_TEMP=0.7 etc.) ----
set(WA_TEMP           "0.85" CACHE STRING "temperature")
set(WA_TOP_K          "40"   CACHE STRING "top-k")
set(WA_TOP_P          "0.95" CACHE STRING "top-p")
set(WA_REPEAT_PENALTY "1.1"  CACHE STRING "repeat penalty")
set(WA_MAX_TOKENS     "80"   CACHE STRING "max new tokens")
set(WA_N_CTX          "1024" CACHE STRING "context size")
set(WA_N_THREADS      "4"    CACHE STRING "CPU threads")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(watcher_jni SHARED llama_jni.cpp)
target_include_directories(watcher_jni PRIVATE llama.cpp/include llama.cpp/ggml/include)
target_compile_definitions(watcher_jni PRIVATE
    WA_TEMP=${WA_TEMP}f
    WA_TOP_K=${WA_TOP_K}
    WA_TOP_P=${WA_TOP_P}f
    WA_REPEAT_PENALTY=${WA_REPEAT_PENALTY}f
    WA_MAX_TOKENS=${WA_MAX_TOKENS}
    WA_N_CTX=${WA_N_CTX}
    WA_N_THREADS=${WA_N_THREADS})
target_link_libraries(watcher_jni llama ggml log)
target_link_options(watcher_jni PRIVATE -Wl,-z,max-page-size=16384)
set_target_properties(watcher_jni PROPERTIES OUTPUT_NAME "llama")
