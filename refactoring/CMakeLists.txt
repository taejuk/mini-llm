cmake_minimum_required(VERSION 3.18)
project(mini_llm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_compile_definitions(WEIGHTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights")
# kernels 라이브러리 생성
add_library(mini_llm_kernels
    src/kernels/gemm.cu
    src/kernels/embedding.cu
    src/kernels/layernorm.cu
    src/kernels/prefill/flashattention.cu
)
target_include_directories(mini_llm_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include    
)
target_link_libraries(mini_llm_kernels PUBLIC CUDA::cudart)

# model 라이브러리 생성
add_library(mini_llm_model
    src/model/gpt2_model.cu
)
target_include_directories(mini_llm_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mini_llm_model PUBLIC
    mini_llm_kernels
    CUDA::cudart
)

# runtime 라이브러리 생성
add_library(mini_llm_runtime
    src/runtime/scheduler.cpp
    src/runtime/server.cpp
)

target_include_directories(mini_llm_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mini_llm_runtime PUBLIC
    mini_llm_model
    Threads::Threads
    uv
)

# 실행파일 생성
add_executable(mini_llm
    src/main.cpp
)

target_include_directories(mini_llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini_llm PRIVATE
    mini_llm_runtime
)

