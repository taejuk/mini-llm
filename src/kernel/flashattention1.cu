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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;

    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    return val;
}

__device__ __forceinline__ float reduce_dhead64_sum(float partial) {
    __shared__ float warp_sums[Br][2];
    int d = threadIdx.x; 
    int r = threadIdx.y;
    int warp_id = d / 32;
    int lane = d % 32;
    float sum = warp_reduce_sum(partial);
    if(lane == 0) warp_sums[r][warp_id] = sum;
    __syncthreads();

    float total = 0.0f;
    if(lane == 0) {
        total = warp_sums[r][0] + warp_sums[r][1];
        warp_sums[r][0] = total;
    }
    __syncthreads();

    return warp_sums[r][0];
}


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
    // O[r][d]요소를 차지한다.
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
    // Q_shared에 저장한다.
    Q_shared[r * d_head + d] = buf_qkv[(q_start + r)*3*d_model + hid * d_head + d];
    __syncthreads();
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float o = 0.0f;

    int q_tile_last = min(q_start + Br - 1, seq_len-1);
    
    for(int k_start = 0; k_start < seq_len; k_start += Bc) {
        // K와 V를 가져온다.
        if(k_start > q_tile_last) break;

        int k_tile_last = min(k_start + Bc - 1, seq_len-1);
        bool full_visible_tile = (k_tile_last <= q_start);    
        int d_head_vec4 = d_head / 4;
        int kv_vec_elems = Bc * d_head_vec4;

        for(int i = tid; i < kv_vec_elems; i+=nthreads) {
            int kk = i / d_head_vec4;
            int d4 = i % d_head_vec4;

            int dd = d4 * 4;
            int k_abs = k_start + kk;

            float4 k4;
            float4 v4;

            if(k_abs < seq_len) {
                const float* k_src = buf_qkv
                    + (size_t)k_abs * 3 * d_model
                    + d_model
                    + hid * d_head
                    + dd;

                const float* v_src = buf_qkv
                    + (size_t)k_abs * 3 * d_model
                    + 2 * d_model
                    + hid * d_head
                    + dd;
                
                k4 = *reinterpret_cast<const float4*>(k_src);
                v4 = *reinterpret_cast<const float4*>(v_src);

            } else {
                k4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            float* k_dst = K_shared + kk * d_head + dd;
            float* v_dst = V_shared + kk * d_head + dd;

            *reinterpret_cast<float4*>(k_dst) = k4;
            *reinterpret_cast<float4*>(v_dst) = v4;
        }
        __syncthreads();
        // QxK^T를 구한다
        for(int kk = 0; kk < Bc; kk++) {
            int k_abs = k_start + kk;
            bool valid_q = q < seq_len;
            bool valid_k = (k_abs < seq_len);

            bool causal;
            if(full_visible_tile) causal = true;
            else causal = k_abs <= q;

            float partial = 0.0f;
	        bool active = valid_q && valid_k && causal;
            if(active && d < d_head)
                partial = Q_shared[r*d_head + d] * K_shared[kk * d_head + d];
            
	        // if (d < d_head) {
            //     partial_shared[r * d_head + d] = partial;
            // }
            // __syncthreads();

            // for (int offset = d_head / 2; offset > 0; offset >>= 1) {
            //     if (d < offset) {
            //         partial_shared[r * d_head + d] +=
            //             partial_shared[r * d_head + d + offset];
            //     }

            //     __syncthreads();
            // }
            float score = reduce_dhead64_sum(partial) * scale;

            if (active && d < d_head) {
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

