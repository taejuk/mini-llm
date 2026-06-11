#include "kernels/decode/paged_attention.cuh"

#include <math_constants.h>
#include <math_functions.h>

#include "constants.h"
#include "kernels/warp_reduction.cuh"

namespace mini_llm::kernels {

__global__ void paged_decode_attention_kernel(
    const float* __restrict__ buf_qkv,          // [B, 3D]
    const int* __restrict__ d_block_table,      // flattened block tables
    const int* __restrict__ d_block_offsets,    // [B]
    const int* __restrict__ d_num_tokens,       // [B]
    const float* __restrict__ pool,
    float* __restrict__ buf_O                   // [B, D]
) {
    constexpr int D = mini_llm::constants::GPT2_D_MODEL;
    constexpr int DH = mini_llm::constants::GPT2_D_HEAD;
    constexpr int BLOCK_SIZE = mini_llm::constants::DEFAULT_KV_BLOCK_SIZE;
    constexpr int TOTAL_BLOCKS = mini_llm::constants::DEFAULT_TOTAL_KV_BLOCKS;

    int b = blockIdx.x;      // request index in batch
    int h = blockIdx.y;      // head index
    int tid = threadIdx.x;

    int num_tokens = d_num_tokens[b];

    const int* block_table = d_block_table + d_block_offsets[b];

    const float* k_pool = pool;
    const float* v_pool =
        pool + static_cast<size_t>(TOTAL_BLOCKS) * BLOCK_SIZE * D;

    const float* q =
        buf_qkv
        + static_cast<size_t>(b) * 3 * D
        + h * DH;

    float* out =
        buf_O
        + static_cast<size_t>(b) * D
        + h * DH;

    if (num_tokens <= 0) {
        for (int d = tid; d < DH; d += blockDim.x) {
            out[d] = 0.0f;
        }
        return;
    }

    extern __shared__ float scores[];

    // ------------------------------------------------------------
    // 1. QK score 계산 + block max
    // ------------------------------------------------------------
    float local_max = -CUDART_INF_F;

    for (int t = tid; t < num_tokens; t += blockDim.x) {
        int logical_block = t / BLOCK_SIZE;
        int offset = t % BLOCK_SIZE;
        int physical_block = block_table[logical_block];

        const float* k =
            k_pool
            + static_cast<size_t>(physical_block) * BLOCK_SIZE * D
            + static_cast<size_t>(offset) * D
            + h * DH;

        float acc = 0.0f;

        #pragma unroll
        for (int d = 0; d < DH; d++) {
            acc += q[d] * k[d];
        }

        float score = acc / sqrtf(static_cast<float>(DH));

        scores[t] = score;
        local_max = fmaxf(local_max, score);
    }

    float max_score = block_reduce_max(local_max);

    // ------------------------------------------------------------
    // 2. exp(score - max) 계산 + block sum
    // ------------------------------------------------------------
    float local_sum = 0.0f;

    for (int t = tid; t < num_tokens; t += blockDim.x) {
        float e = expf(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }

    float denom = block_reduce_sum(local_sum) + 1e-6f;

    // ------------------------------------------------------------
    // 3. softmax(score) * V
    // ------------------------------------------------------------
    for (int d = tid; d < DH; d += blockDim.x) {
        float acc = 0.0f;

        for (int t = 0; t < num_tokens; t++) {
            int logical_block = t / BLOCK_SIZE;
            int offset = t % BLOCK_SIZE;
            int physical_block = block_table[logical_block];

            const float* v =
                v_pool
                + static_cast<size_t>(physical_block) * BLOCK_SIZE * D
                + static_cast<size_t>(offset) * D
                + h * DH;

            acc += scores[t] * v[d];
        }

        out[d] = acc / denom;
    }
}

void paged_decode_attention(
    const float* buf_qkv,
    const int* d_block_table,
    const int* d_block_offsets,
    const int* d_num_tokens,
    const float* pool,
    float* buf_O,
    int batch_size,
    int max_seq
) {
    if (batch_size <= 0) {
        return;
    }

    dim3 grid(
        batch_size,
        mini_llm::constants::GPT2_N_HEADS
    );

    dim3 block(256);

    size_t shared_bytes =
        static_cast<size_t>(max_seq) * sizeof(float);

    paged_decode_attention_kernel<<<grid, block, shared_bytes>>>(
        buf_qkv,
        d_block_table,
        d_block_offsets,
        d_num_tokens,
        pool,
        buf_O
    );
}

} // namespace mini_llm::kernels