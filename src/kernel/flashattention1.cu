#include "kernel/flashattention1.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef FA1_BLOCK_M
#define FA1_BLOCK_M 4
#endif

#ifndef FA1_BLOCK_N
#define FA1_BLOCK_N 32
#endif

#define Br FA1_BLOCK_M
#define Bc FA1_BLOCK_N

#define CUDA_CHECK_FLASH(call)                                             \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void flashattention1_prefill_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ buf_O,
    int seq_len,
    int d_model,
    int d_head,
    float scale
) {
    int hid = blockIdx.x;
    int q_tile = blockIdx.y;

    int d = threadIdx.x; 
    int r = threadIdx.y;  

    int q_start = q_tile * Br;
    int q = q_start + r;

    extern __shared__ float smem[];

    float* Q_shared = smem;
    float* K_shared = Q_shared + Br * d_head;
    float* V_shared = K_shared + Bc * d_head;
    float* partial_shared = V_shared + Bc * d_head;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    /*
     * 1. Q tile load: [Br, d_head]
     */
    int q_elems = Br * d_head;

    for (int idx = tid; idx < q_elems; idx += nthreads) {
        int rr = idx / d_head;
        int dd = idx % d_head;

        int q_abs = q_start + rr;

        float val = 0.0f;

        if (q_abs < seq_len) {
            val = buf_qkv[
                (size_t)q_abs * 3 * d_model
                + hid * d_head
                + dd
            ];
        }

        Q_shared[idx] = val;
    }

    __syncthreads();

    /*
     * thread 하나가 O[q, hid, d] 하나의 accumulator를 담당
     */
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float o = 0.0f;

    float q_val = 0.0f;

    if (q < seq_len && d < d_head) {
        q_val = Q_shared[r * d_head + d];
    }

    /*
     * 2. K/V tile loop
     */
    for (int k_start = 0; k_start < seq_len; k_start += Bc) {
        /*
         * 2-1. K/V tile load: [Bc, d_head]
         */
        int kv_elems = Bc * d_head;

        for (int idx = tid; idx < kv_elems; idx += nthreads) {
            int kk = idx / d_head;
            int dd = idx % d_head;

            int k_abs = k_start + kk;

            float kval = 0.0f;
            float vval = 0.0f;

            if (k_abs < seq_len) {
                kval = buf_qkv[
                    (size_t)k_abs * 3 * d_model
                    + d_model
                    + hid * d_head
                    + dd
                ];

                vval = buf_qkv[
                    (size_t)k_abs * 3 * d_model
                    + 2 * d_model
                    + hid * d_head
                    + dd
                ];
            }

            K_shared[idx] = kval;
            V_shared[idx] = vval;
        }

        __syncthreads();

        /*
         * 2-2. 현재 K tile 안의 key들을 순회
         */
        for (int kk = 0; kk < Bc; kk++) {
            int k_abs = k_start + kk;

            bool valid = (q < seq_len) && (k_abs < seq_len) && (k_abs <= q);

            /*
             * score = dot(Q[q], K[k])
             */
            float partial = 0.0f;

            if (valid && d < d_head) {
                partial = q_val * K_shared[kk * d_head + d];
            }

            partial_shared[r * d_head + d] = partial;

            __syncthreads();

            /*
             * reduce over d_head
             */
            // 이거를 warp reduction으로 최적화하자.
            for (int offset = d_head / 2; offset > 0; offset >>= 1) {
                if (d < offset) {
                    partial_shared[r * d_head + d] +=
                        partial_shared[r * d_head + d + offset];
                }
                __syncthreads();
            }

            float score = partial_shared[r * d_head] * scale;

            /*
             * online softmax update
             */
            if (valid && d < d_head) {
                float m_new = fmaxf(m, score);
                float alpha = expf(m - m_new);
                float p = expf(score - m_new);

                float v_val = V_shared[kk * d_head + d];

                o = o * alpha + p * v_val;
                l = l * alpha + p;
                m = m_new;
            }

            __syncthreads();
        }

        __syncthreads();
    }

    /*
     * 3. Write O
     */
    if (q < seq_len && d < d_head) {
        buf_O[
            (size_t)q * d_model
            + hid * d_head
            + d
        ] = o / l;
    }
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

    dim3 grid(n_heads, (seq_len + Br - 1) / Br);
    dim3 block(d_head, Br);

    size_t smem_size =
        ((size_t)Br * d_head +        // Q_shared
         (size_t)Bc * d_head +        // K_shared
         (size_t)Bc * d_head +        // V_shared
         (size_t)Br * d_head)         // partial_shared
        * sizeof(float);

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
