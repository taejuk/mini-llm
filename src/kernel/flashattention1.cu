#include "kernel/flashattention1.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CUDA_CHECK_FLASH(call)                                             \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr,                                                 \
                    "CUDA error %s:%d: %s\n",                               \
                    __FILE__,                                               \
                    __LINE__,                                               \
                    cudaGetErrorString(e));                                 \
            exit(1);                                                        \
        }                                                                  \
    } while (0)

/*
 * FlashAttention1-style prefill kernel.
 *
 * 목적:
 *   기존 prefill attention:
 *     QK^T -> causal mask -> softmax -> S @ V
 *
 *   를 하나의 online-softmax kernel로 대체한다.
 *
 * 입력:
 *   buf_qkv: [seq_len, 3 * d_model]
 *            row = [Q | K | V]
 *
 * 출력:
 *   buf_O: [seq_len, d_model]
 *
 * mapping:
 *   grid.x = head index
 *   grid.y = query token index
 *   threadIdx.x = head dimension index
 *
 * 주의:
 *   이 코드는 성능 비교용 FlashAttention1 baseline이다.
 *   production-grade FlashAttention처럼 BLOCK_M x BLOCK_N tile을
 *   고도로 최적화한 버전은 아니다.
 */
__global__ void flashattention1_prefill_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ buf_O,
    int seq_len,
    int d_model,
    int d_head,
    float scale
) {
    int h   = blockIdx.x;
    int qi  = blockIdx.y;
    int tid = threadIdx.x;

    if (tid >= d_head) return;

    extern __shared__ float smem[];

    const float* q_ptr =
        buf_qkv
        + (size_t)qi * 3 * d_model
        + h * d_head;

    float q_val = q_ptr[tid];

    /*
     * Online softmax state.
     *
     * m: 현재까지 본 score의 max
     * l: 현재까지 본 exp sum
     * o: 현재 thread가 담당하는 output dimension 누적값
     */
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float o = 0.0f;

    /*
     * Causal attention:
     * query qi는 key 0..qi까지만 볼 수 있다.
     */
    for (int kj = 0; kj <= qi; kj++) {
        const float* k_ptr =
            buf_qkv
            + (size_t)kj * 3 * d_model
            + d_model
            + h * d_head;

        const float* v_ptr =
            buf_qkv
            + (size_t)kj * 3 * d_model
            + 2 * d_model
            + h * d_head;

        /*
         * score = dot(Q[qi,h], K[kj,h]) * scale
         *
         * 각 thread가 head dimension 하나의 partial product를 계산하고,
         * block 내부 reduction으로 dot product를 만든다.
         */
        float partial = q_val * k_ptr[tid];
        smem[tid] = partial;
        __syncthreads();

        /*
         * d_head가 power of 2라고 가정.
         * GPT-2 small에서는 d_head = 64.
         */
        for (int offset = d_head / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                smem[tid] += smem[tid + offset];
            }
            __syncthreads();
        }

        float score = smem[0] * scale;

        /*
         * Online softmax update.
         *
         * m_new = max(m_old, score)
         * l_new = exp(m_old - m_new) * l_old
         *       + exp(score - m_new)
         *
         * o_new = exp(m_old - m_new) * o_old
         *       + exp(score - m_new) * V[kj]
         */
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p     = expf(score - m_new);

        o = o * alpha + p * v_ptr[tid];
        l = l * alpha + p;
        m = m_new;

        __syncthreads();
    }

    float out = o / l;

    buf_O[
        (size_t)qi * d_model
        + h * d_head
        + tid
    ] = out;
}

void flashattention1_prefill(
    const float* buf_qkv,
    float* buf_O,
    int seq_len,
    int d_model,
    int d_head,
    int n_heads,
    float scale
) {
    if (seq_len <= 0) return;

    dim3 grid(n_heads, seq_len);
    dim3 block(d_head);

    size_t smem_size = (size_t)d_head * sizeof(float);

    flashattention1_prefill_kernel<<<grid, block, smem_size>>>(
        buf_qkv,
        buf_O,
        seq_len,
        d_model,
        d_head,
        scale
    );

    CUDA_CHECK_FLASH(cudaGetLastError());
}
