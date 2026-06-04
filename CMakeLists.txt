cmake_minimum_required(VERSION 3.18)
project(paged_attention LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")



add_compile_definitions(WEIGHTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/weights")

# ── vllm 라이브러리 (BlockAllocator + PagedKVCache) ──────────────
add_library(vllm_lib
    src/vllm/allocator.cu
    src/vllm/kv_cache.cu
)
target_include_directories(vllm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(vllm_lib PUBLIC CUDA::cudart)

# ── GPT-2 공용 커널 ──────────────────────────────────────────────
add_library(gpt2_common_lib
    src/model/gpt2_common.cu
)
target_include_directories(gpt2_common_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gpt2_common_lib PUBLIC vllm_lib CUDA::cublas)

# -- HF - DEBUG 모드를 추가한다.
option(ENABLE_TENSOR_DUMP "Enable tensor dump for correctness debugging" OFF)

add_library(tensor_dump_lib
    src/debug/tensor_dumper.cpp
)

target_include_directories(tensor_dump_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tensor_dump_lib PUBLIC CUDA::cudart)

if (ENABLE_TENSOR_DUMP)
    target_compile_definitions(tensor_dump_lib PUBLIC ENABLE_TENSOR_DUMP)
endif()


# ── GPT-2 WMMA (Tensor Core) ─────────────────────────────────────
set(USE_FLASH_ATTENTION_PREFILL "0" CACHE STRING "Use FlashAttention1-style prefill attention")
set(FA1_BLOCK_M "4" CACHE STRING "FlashAttention1 query tile size")
set(FA1_BLOCK_N "32" CACHE STRING "FlashAttention1 key/value tile size")

add_library(gpt2_wmma_lib
    src/model/gpt2_wmma.cu
    src/kernel/wmma_gemm.cu
    src/kernel/flashattention1.cu
)

target_include_directories(gpt2_wmma_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(gpt2_wmma_lib PRIVATE
    USE_FLASH_ATTENTION_PREFILL=${USE_FLASH_ATTENTION_PREFILL}
    FA1_BLOCK_M=${FA1_BLOCK_M}
    FA1_BLOCK_N=${FA1_BLOCK_N}
)

if (ENABLE_TENSOR_DUMP)
    target_compile_definitions(gpt2_wmma_lib PUBLIC ENABLE_TENSOR_DUMP)
endif()

target_link_libraries(gpt2_wmma_lib PUBLIC
    gpt2_common_lib
    tensor_dump_lib
)
# ── GPT-2 FP32 모델 ──────────────────────────────────────────────
add_library(gpt2_fp32_lib
    src/model/gpt2_model.cu
)
target_include_directories(gpt2_fp32_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gpt2_fp32_lib PUBLIC gpt2_common_lib)

# ── Scheduler ────────────────────────────────────────────────────
# 기존: test에 scheduler.cpp 직접 컴파일
# 변경: 라이브러리로 분리 → 서버와 테스트가 공유
add_library(scheduler_lib
    src/scheduler/scheduler.cpp
)
target_include_directories(scheduler_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(scheduler_lib PUBLIC vllm_lib)

# ── HTTP 서버 ─────────────────────────────────────────────────────
add_library(http_server_lib
    src/server/http_server.cpp
)
target_include_directories(http_server_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# eventfd, epoll → Linux 커널 기본 제공, 추가 링크 불필요

# ── 서버 실행 파일 ───────────────────────────────────────────────
# main.cu는 CUDA 코드 없지만 .cu 확장자 → nvcc가 host 코드로 처리
add_executable(gpt2_server
    src/main.cu
)
target_include_directories(gpt2_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gpt2_server PRIVATE
    http_server_lib
    scheduler_lib
    gpt2_wmma_lib
    gpt2_fp32_lib
)

# ── GPT-2 WMMA token latency benchmark ─────────────────────────
add_executable(bench_gpt2_wmma_token_latency
    benchmarks/gpt2/bench_gpt2_wmma_token_latency.cu
)
target_include_directories(bench_gpt2_wmma_token_latency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(bench_gpt2_wmma_token_latency PRIVATE
    gpt2_wmma_lib
)

# ── FlashAttention1 only benchmark ───────────────────────────────
add_executable(bench_flashattention1_only
    benchmarks/gpt2/bench_flashattention1_only.cu
)

target_include_directories(bench_flashattention1_only PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(bench_flashattention1_only PRIVATE
    FA1_BLOCK_M=${FA1_BLOCK_M}
    FA1_BLOCK_N=${FA1_BLOCK_N}
)

target_link_libraries(bench_flashattention1_only PRIVATE
    gpt2_wmma_lib
    CUDA::cudart
)


# ── Unit Tests ───────────────────────────────────────────────────
enable_testing()

add_executable(test_mpsc_queue
    test/test_mpsc_queue.cpp
)
target_include_directories(test_mpsc_queue PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
add_test(NAME test_mpsc_queue COMMAND test_mpsc_queue)

add_executable(test_allocator
    test/test_allocator.cu
)
target_include_directories(test_allocator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(test_allocator PRIVATE vllm_lib)
add_test(NAME test_allocator COMMAND test_allocator)

add_executable(test_kv_cache
    test/test_kv_cache.cu
)
target_include_directories(test_kv_cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(test_kv_cache PRIVATE vllm_lib)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

add_executable(test_scheduler
    test/test_scheduler.cu
)
target_include_directories(test_scheduler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(test_scheduler PRIVATE scheduler_lib CUDA::cudart)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_flashattention1_correctness
    test/test_flashattention1_correctness.cu
)

target_include_directories(test_flashattention1_correctness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_link_libraries(test_flashattention1_correctness PRIVATE
    gpt2_wmma_lib
    CUDA::cudart
)

add_test(NAME test_flashattention1_correctness COMMAND test_flashattention1_correctness)

# ── Offline GPT-2 WMMA serving benchmark ───────────────────────
add_executable(bench_offline_gpt2_wmma
    benchmarks/serving/offline_gpt2_wmma_bench.cu
)

target_include_directories(bench_offline_gpt2_wmma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bench_offline_gpt2_wmma PRIVATE
    scheduler_lib
    gpt2_wmma_lib
    CUDA::cudart
)

# ── Offline GPT-2 WMMA serving benchmark: decode no-batch ──────
add_executable(bench_offline_gpt2_wmma_decode_nobatch
    benchmarks/serving/offline_gpt2_wmma_decode_nobatch_bench.cu
)

target_include_directories(bench_offline_gpt2_wmma_decode_nobatch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bench_offline_gpt2_wmma_decode_nobatch PRIVATE
    scheduler_lib
    gpt2_wmma_lib
    CUDA::cudart
)


# ── GPT-2 WMMA prefill prompt length sweep benchmark ─────────────
add_executable(bench_gpt2_wmma_prefill_sweep
    benchmarks/gpt2/bench_gpt2_wmma_prefill_sweep.cu
)

target_include_directories(bench_gpt2_wmma_prefill_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bench_gpt2_wmma_prefill_sweep PRIVATE
    gpt2_wmma_lib
    CUDA::cudart
)

add_executable(test_wmma_vs_hf
    test/test_wmma_vs_hf.cu
)

target_include_directories(test_wmma_vs_hf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_link_libraries(test_wmma_vs_hf PRIVATE
    gpt2_wmma_lib
    tensor_dump_lib
    CUDA::cudart
)

if (ENABLE_TENSOR_DUMP)
    target_compile_definitions(test_wmma_vs_hf PRIVATE ENABLE_TENSOR_DUMP)
endif()

add_test(NAME test_wmma_vs_hf COMMAND test_wmma_vs_hf)