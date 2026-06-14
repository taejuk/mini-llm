cmake_minimum_required(VERSION 3.18)
project(mini_llm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

option(BUILD_RUNTIME "Build runtime/server" OFF)

add_compile_definitions(WEIGHTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights")
# kernels 라이브러리 생성
add_library(mini_llm_kernels
    src/kernels/gemm.cu
    src/kernels/linear.cu
    src/kernels/embedding.cu
    src/kernels/layernorm.cu
    src/kernels/residual.cu
    src/kernels/gelu.cu
    src/kernels/prefill/append_kv.cu
    src/kernels/prefill/flashattention.cu
    src/kernels/decode/paged_attention.cu
)

target_include_directories(mini_llm_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include    
)
target_link_libraries(mini_llm_kernels PUBLIC CUDA::cudart)

# model 라이브러리 생성
add_library(mini_llm_model
    src/model/gpt2_model.cu
    src/model/gpt2_test_api.cu
)
target_include_directories(mini_llm_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(mini_llm_model PUBLIC
    mini_llm_kernels
    CUDA::cudart
)

# runtime 라이브러리 생성
if(BUILD_RUNTIME)

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
endif()
# 테스트 파일 생성
enable_testing()

add_compile_definitions(TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels")

function(add_kernel_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
    )
    target_link_libraries(${name} PRIVATE
        mini_llm_kernels
        CUDA::cudart
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

function(add_model_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp
    )
    target_compile_definitions(${name} PRIVATE
        MODEL_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/model"
    )
    target_link_libraries(${name} PRIVATE
        mini_llm_model
        CUDA::cudart
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_kernel_test(test_residual tests/cpp/test_residual.cu)
add_kernel_test(test_gelu tests/cpp/test_gelu.cu)
add_kernel_test(test_layernorm tests/cpp/test_layernorm.cu)
add_kernel_test(test_gemm tests/cpp/test_gemm.cu)
add_kernel_test(test_embedding tests/cpp/test_embedding.cu)
add_kernel_test(test_append_kv tests/cpp/test_append_kv.cu)
add_kernel_test(test_flashattention tests/cpp/test_flashattention.cu)

add_model_test(test_gpt2_prefill_logits tests/cpp/test_gpt2_prefill_logits.cu)
