#include "test_common.cuh"

#include "constants.h"
#include "kernels/prefill/append_kv.cuh"

int main() {
    namespace C = mini_llm::constants;

    constexpr int SEQ = 6;
    constexpr int D = C::GPT2_D_MODEL;
    constexpr int BLOCK_SIZE = C::DEFAULT_KV_BLOCK_SIZE;
    constexpr int TOTAL_BLOCKS = C::DEFAULT_TOTAL_KV_BLOCKS;

    auto h_qkv = read_bin<float>(case_path("append_kv", "buf_qkv.bin"), SEQ * 3 * D);
    auto h_token_to_block = read_bin<int>(case_path("append_kv", "token_to_block.bin"), SEQ);
    auto h_token_to_offset = read_bin<int>(case_path("append_kv", "token_to_offset.bin"), SEQ);
    auto expected_k = read_bin<float>(case_path("append_kv", "expected_k.bin"), SEQ * D);
    auto expected_v = read_bin<float>(case_path("append_kv", "expected_v.bin"), SEQ * D);

    float* d_qkv = nullptr;
    float* d_pool = nullptr;
    int* d_token_to_block = nullptr;
    int* d_token_to_offset = nullptr;

    size_t pool_numel =
        static_cast<size_t>(2) * TOTAL_BLOCKS * BLOCK_SIZE * D;

    CUDA_CHECK(cudaMalloc(&d_qkv, SEQ * 3 * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool, pool_numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_token_to_block, SEQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_token_to_offset, SEQ * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_pool, 0, pool_numel * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_qkv, h_qkv.data(), SEQ * 3 * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_token_to_block, h_token_to_block.data(), SEQ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_token_to_offset, h_token_to_offset.data(), SEQ * sizeof(int), cudaMemcpyHostToDevice));

    mini_llm::kernels::append_prefill_kv(
        d_qkv,
        d_pool,
        d_token_to_block,
        d_token_to_offset,
        SEQ
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got_k(SEQ * D);
    std::vector<float> got_v(SEQ * D);

    float* d_k_pool = d_pool;
    float* d_v_pool = d_pool + static_cast<size_t>(TOTAL_BLOCKS) * BLOCK_SIZE * D;

    for (int t = 0; t < SEQ; ++t) {
        int block_id = h_token_to_block[t];
        int offset = h_token_to_offset[t];

        size_t base =
            static_cast<size_t>(block_id) * BLOCK_SIZE * D +
            static_cast<size_t>(offset) * D;

        CUDA_CHECK(cudaMemcpy(
            got_k.data() + t * D,
            d_k_pool + base,
            D * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaMemcpy(
            got_v.data() + t * D,
            d_v_pool + base,
            D * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
    }

    check_close("append_kv K", got_k, expected_k, 1e-6f);
    check_close("append_kv V", got_v, expected_v, 1e-6f);

    cudaFree(d_qkv);
    cudaFree(d_pool);
    cudaFree(d_token_to_block);
    cudaFree(d_token_to_offset);

    return 0;
}