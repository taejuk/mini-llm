cmake_minimum_required(VERSION 3.18)

project(mini_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_RUNTIME "Build runtime/server" OFF)
option(USE_MOCK_BACKEND "Use mock backend for runtime/server" OFF)
option(BUILD_CUDA_TESTS "Build CUDA kernel/model tests" ON)

find_package(Threads REQUIRED)

set(NEED_CUDA OFF)

if(BUILD_CUDA_TESTS)
    set(NEED_CUDA ON)
endif()

if(BUILD_RUNTIME AND NOT USE_MOCK_BACKEND)
    set(NEED_CUDA ON)
endif()

if(NEED_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

    add_compile_definitions(
        WEIGHTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights"
    )

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

    target_link_libraries(mini_llm_kernels PUBLIC
        CUDA::cudart
    )

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
endif()

if(BUILD_RUNTIME)
    find_path(UV_INCLUDE_DIR uv.h)
    find_library(UV_LIBRARY uv)

    if(NOT UV_INCLUDE_DIR)
        message(FATAL_ERROR "uv.h not found. Install libuv or set UV_INCLUDE_DIR.")
    endif()

    if(NOT UV_LIBRARY)
        message(FATAL_ERROR "libuv not found. Install libuv or set UV_LIBRARY.")
    endif()

    add_library(mini_llm_runtime
        src/runtime/scheduler.cpp
        src/runtime/server.cpp
    )

    target_include_directories(mini_llm_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${UV_INCLUDE_DIR}
    )

    target_link_libraries(mini_llm_runtime PUBLIC
        Threads::Threads
        ${UV_LIBRARY}
    )

    if(USE_MOCK_BACKEND)
        target_compile_definitions(mini_llm_runtime PUBLIC
            MINI_LLM_USE_MOCK_BACKEND=1
        )
    else()
        target_compile_definitions(mini_llm_runtime PUBLIC
            MINI_LLM_USE_MOCK_BACKEND=0
        )

        target_link_libraries(mini_llm_runtime PUBLIC
            mini_llm_model
            CUDA::cudart
        )
    endif()

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

if(BUILD_CUDA_TESTS)
    if(NOT NEED_CUDA)
        message(FATAL_ERROR "BUILD_CUDA_TESTS=ON requires CUDA.")
    endif()

    enable_testing()

    add_compile_definitions(
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels"
    )

    function(add_kernel_test name src)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
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
        endif()
    endfunction()

    function(add_model_test name src)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
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
        endif()
    endfunction()

    add_kernel_test(test_residual tests/cpp/test_residual.cu)
    add_kernel_test(test_gelu tests/cpp/test_gelu.cu)
    add_kernel_test(test_layernorm tests/cpp/test_layernorm.cu)
    add_kernel_test(test_gemm tests/cpp/test_gemm.cu)
    add_kernel_test(test_embedding tests/cpp/test_embedding.cu)
    add_kernel_test(test_append_kv tests/cpp/test_append_kv.cu)
    add_kernel_test(test_flashattention tests/cpp/test_flashattention.cu)

    add_model_test(test_gpt2_prefill_logits tests/cpp/test_gpt2_prefill_logits.cu)
    add_model_test(test_gpt2_decode_logits tests/cpp/test_gpt2_decode_logits.cu)
endif()