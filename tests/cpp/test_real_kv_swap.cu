#include "constants.h"
#include "runtime/block_manager.h"
#include "runtime/cpu_pool.cuh"
#include "runtime/pool.cuh"
#include "runtime/real_kv_allocator.h"
#include "runtime/request.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(err) << "\n";        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

#define CHECK_TRUE(cond, msg)                                              \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "[FAIL] " << msg << "\n";                         \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace {

__host__ __device__ float expected_value(
    int layer,
    int logical_block,
    int elem
) {
    return static_cast<float>(
        layer * 100000 +
        logical_block * 1000 +
        (elem % 997)
    );
}

__global__ void fill_kv_block_kernel(
    float* pool,
    int physical_block_id,
    int layer,
    int logical_block,
    size_t block_elems
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= block_elems) {
        return;
    }

    pool[
        static_cast<size_t>(physical_block_id) * block_elems + tid
    ] = expected_value(
        layer,
        logical_block,
        static_cast<int>(tid)
    );
}

std::vector<int> make_prompt(int token_len) {
    std::vector<int> base = {15496, 11, 616, 1438, 318};
    std::vector<int> prompt;
    prompt.reserve(token_len);

    for (int i = 0; i < token_len; ++i) {
        prompt.push_back(base[i % static_cast<int>(base.size())]);
    }

    return prompt;
}

int valid_tokens_for_block(int prompt_len, int logical_block) {
    int block_start = logical_block * C::DEFAULT_KV_BLOCK_SIZE;
    int remain = prompt_len - block_start;

    if (remain <= 0) {
        return 0;
    }

    return std::min(remain, C::DEFAULT_KV_BLOCK_SIZE);
}

void fill_request_kv_data(
    Rt::Request& req,
    Rt::Pool& pool
) {
    size_t block_elems = pool.block_slot_size();

    constexpr int threads = 256;
    int grid = static_cast<int>(
        (block_elems + threads - 1) / threads
    );

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];

        for (int b = 0; b < kv.num_blocks(); ++b) {
            int physical_block_id = kv.block_table_[b];

            fill_kv_block_kernel<<<grid, threads>>>(
                pool.pool_start(),
                physical_block_id,
                layer,
                b,
                block_elems
            );

            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void verify_request_kv_data(
    Rt::Request& req,
    Rt::Pool& pool
) {
    size_t block_elems = pool.block_slot_size();
    size_t block_bytes = block_elems * sizeof(float);

    std::vector<float> h_block(block_elems);

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];

        for (int b = 0; b < kv.num_blocks(); ++b) {
            int physical_block_id = kv.block_table_[b];

            CUDA_CHECK(cudaMemcpy(
                h_block.data(),
                pool.block_ptr(physical_block_id),
                block_bytes,
                cudaMemcpyDeviceToHost
            ));

            for (size_t e = 0; e < block_elems; ++e) {
                float expected = expected_value(
                    layer,
                    b,
                    static_cast<int>(e)
                );

                float got = h_block[e];

                if (std::fabs(got - expected) > 1e-5f) {
                    std::cerr
                        << "[FAIL] KV data mismatch"
                        << " layer=" << layer
                        << " logical_block=" << b
                        << " elem=" << e
                        << " expected=" << expected
                        << " got=" << got
                        << "\n";
                    std::exit(1);
                }
            }
        }
    }
}

void set_request_num_tokens(
    Rt::Request& req,
    int prompt_len
) {
    for (auto& kv : req.layer_kv) {
        kv.set_num_tokens(prompt_len);
    }
}

void verify_restored_metadata(
    Rt::Request& req,
    Rt::BlockManager& block_manager,
    int prompt_len,
    int blocks_per_layer
) {
    CHECK_TRUE(
        req.kv_residency == Rt::KvCacheResidency::Gpu,
        "request kv_residency should be Gpu after swap_in"
    );

    CHECK_TRUE(
        req.state == Rt::RequestState::DecodeReady,
        "request state should be DecodeReady after swap_in"
    );

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];

        CHECK_TRUE(
            kv.num_tokens_ == prompt_len,
            "PagedKVCache num_tokens_ was not restored"
        );

        CHECK_TRUE(
            kv.num_blocks() == blocks_per_layer,
            "PagedKVCache block count was not restored"
        );

        for (int b = 0; b < blocks_per_layer; ++b) {
            int physical_block_id = kv.block_table_[b];

            int expected_filled = valid_tokens_for_block(
                prompt_len,
                b
            );

            int got_filled =
                block_manager.block(physical_block_id).filled();

            CHECK_TRUE(
                got_filled == expected_filled,
                "PhysicalBlock filled count was not restored"
            );
        }
    }
}

void test_swap_out_in_restores_state_and_data() {
    Rt::Pool& pool =
        Rt::Pool::getInstance(C::DEFAULT_TOTAL_KV_BLOCKS);

    Rt::CpuPool& cpu_pool =
        Rt::CpuPool::getInstance(C::DEFAULT_TOTAL_CPU_BLOCKS);

    Rt::BlockManager& block_manager =
        Rt::BlockManager::getInstance(pool, cpu_pool);

    Rt::RealKvAllocator allocator;

    int prompt_len = 37;
    int max_new_tokens = 8;

    int blocks_per_layer =
        (prompt_len + C::DEFAULT_KV_BLOCK_SIZE - 1) /
        C::DEFAULT_KV_BLOCK_SIZE;

    Rt::Request req(
        1,
        make_prompt(prompt_len),
        max_new_tokens
    );

    CHECK_TRUE(
        allocator.allocate_prefill(req),
        "allocate_prefill failed"
    );

    CHECK_TRUE(
        req.kv_residency == Rt::KvCacheResidency::Gpu,
        "request should be Gpu after allocate_prefill"
    );

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        CHECK_TRUE(
            req.layer_kv[layer].num_blocks() == blocks_per_layer,
            "allocate_prefill allocated wrong number of blocks"
        );
    }

    set_request_num_tokens(req, prompt_len);
    fill_request_kv_data(req, pool);

    CHECK_TRUE(
        allocator.swap_out(req),
        "swap_out failed"
    );

    CHECK_TRUE(
        req.kv_residency == Rt::KvCacheResidency::Cpu,
        "request kv_residency should be Cpu after swap_out"
    );

    CHECK_TRUE(
        req.state == Rt::RequestState::SwappedOut,
        "request state should be SwappedOut after swap_out"
    );

    CHECK_TRUE(
        allocator.is_swapped(req),
        "allocator.is_swapped(req) should be true after swap_out"
    );

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        CHECK_TRUE(
            req.layer_kv[layer].num_blocks() == 0,
            "GPU block table should be cleared after swap_out"
        );

        CHECK_TRUE(
            req.layer_kv[layer].num_tokens_ == 0,
            "PagedKVCache num_tokens_ should be cleared after swap_out"
        );
    }

    CHECK_TRUE(
        allocator.swap_in(req),
        "swap_in failed"
    );

    CHECK_TRUE(
        !allocator.is_swapped(req),
        "allocator.is_swapped(req) should be false after swap_in"
    );

    verify_restored_metadata(
        req,
        block_manager,
        prompt_len,
        blocks_per_layer
    );

    verify_request_kv_data(req, pool);

    allocator.free_request(req);

    std::cout
        << "[PASS] swap_out/swap_in restores request state, "
        << "PagedKVCache metadata, PhysicalBlock metadata, and KV data\n";
}

} // namespace

int main() {
    test_swap_out_in_restores_state_and_data();

    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[PASS] real KV swap tests\n";
    return 0;
}