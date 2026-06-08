#include "kernels/prefill/append_kv.cuh"
#include "constants.h"

namespace mini_llm::kernels {

__global__ void append_prefill_kv_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ k_pool,
    float* __restrict__ v_pool,
    const int* __restrict__ d_token_to_block,
    const int* __restrict__ d_token_to_offset,
    int seq_len
) {
    constexpr int D = mini_llm::constants::GPT2_D_MODEL;
    constexpr int BLOCK_SIZE = mini_llm::constants::DEFAULT_KV_BLOCK_SIZE;
    constexpr int VEC = 4;
    constexpr int VEC_PER_TOKEN = D / VEC;

    static_assert(D % VEC == 0, "D_MODEL must be divisible by 4");

    int row = blockIdx.x;     // token index in buf_qkv
    int col = threadIdx.x;    // float4 index

    if (row >= seq_len || col >= VEC_PER_TOKEN) {
        return;
    }

    int physical_block_id = d_token_to_block[row];
    int offset_in_block = d_token_to_offset[row];

    const float* qkv_row = buf_qkv + row * 3 * D;
    const float* k_src = qkv_row + D;
    const float* v_src = qkv_row + 2 * D;

    int dst_base =
        physical_block_id * BLOCK_SIZE * D +
        offset_in_block * D;

    const float4* k_src4 = reinterpret_cast<const float4*>(k_src);
    const float4* v_src4 = reinterpret_cast<const float4*>(v_src);

    float4* k_dst4 = reinterpret_cast<float4*>(k_pool + dst_base);
    float4* v_dst4 = reinterpret_cast<float4*>(v_pool + dst_base);

    k_dst4[col] = k_src4[col];
    v_dst4[col] = v_src4[col];
}

void append_prefill_kv(
    const float* buf_qkv,
    float* pool,
    const int* d_token_to_block,
    const int* d_token_to_offset,
    int seq_len
) {
    if (seq_len <= 0) {
        return;
    }

    constexpr int D = mini_llm::constants::GPT2_D_MODEL;
    constexpr int BLOCK_SIZE = mini_llm::constants::DEFAULT_KV_BLOCK_SIZE;
    constexpr int VEC = 4;
    constexpr int VEC_PER_TOKEN = D / VEC;

    static_assert(D % VEC == 0, "D_MODEL must be divisible by 4");

    dim3 grid(seq_len);
    dim3 block(VEC_PER_TOKEN);

    
    float* k_pool = pool;
    float* v_pool =
        pool + mini_llm::constants::DEFAULT_TOTAL_KV_BLOCKS * BLOCK_SIZE * D;

    append_prefill_kv_kernel<<<grid, block>>>(
        buf_qkv,
        k_pool,
        v_pool,
        d_token_to_block,
        d_token_to_offset,
        seq_len
    );
}

} // namespace mini_llm::kernels