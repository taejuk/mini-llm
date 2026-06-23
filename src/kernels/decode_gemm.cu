#include "kernels/decode_gemm.cuh"
#include "constants.h"

#include <cuda_runtime.h>
#include <math.h>

namespace mini_llm::kernels {

namespace {

constexpr int DECODE_M_TILE = 1;
constexpr int DECODE_N_TILE = 32;
constexpr int DECODE_K_TILE = 64;

/*
    Decode small-M GEMM:

        C[M, N] = alpha * A[M, K] x B[K, N] + beta * C[M, N] + bias[N]

    Target shapes:
        qkv_projection      : M < 8, K = 768,  N = 2304
        attn_out_projection : M < 8, K = 768,  N = 768
        fc1                 : M < 8, K = 768,  N = 3072
        fc2                 : M < 8, K = 3072, N = 768

    One CUDA block computes:
        C tile [1 row, 32 cols]

    Thread layout:
        blockDim.x = 32  -> output column
        blockDim.y = 1   -> batch row
*/

__device__ __forceinline__ float gelu_one(float v) {
    return 0.5f * v *
           (1.0f + tanhf(
               0.7978845608f *
               (v + 0.044715f * v * v * v)
           ));
}

__global__ void decode_small_m_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    const float* __restrict__ bias,
    float* __restrict__ C,
    bool apply_gelu
) {
    int local_col = threadIdx.x; // 0..31
    int local_row = threadIdx.y; // 0

    int row = blockIdx.y * DECODE_M_TILE + local_row;
    int col = blockIdx.x * DECODE_N_TILE + local_col;

    __shared__ float sA[DECODE_M_TILE][DECODE_K_TILE + 1];
    __shared__ float sB[DECODE_K_TILE][DECODE_N_TILE + 1];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += DECODE_K_TILE) {
        int a_elems = DECODE_M_TILE * DECODE_K_TILE;

        for (int idx = tid; idx < a_elems; idx += num_threads) {
            int r = idx / DECODE_K_TILE;
            int k = idx % DECODE_K_TILE;

            int global_row = blockIdx.y * DECODE_M_TILE + r;
            int global_k = k0 + k;

            if (global_row < M && global_k < K) {
                sA[r][k] =
                    A[static_cast<size_t>(global_row) *
                      static_cast<size_t>(K) +
                      static_cast<size_t>(global_k)];
            } else {
                sA[r][k] = 0.0f;
            }
        }

        int b_elems = DECODE_K_TILE * DECODE_N_TILE;

        for (int idx = tid; idx < b_elems; idx += num_threads) {
            int k = idx / DECODE_N_TILE;
            int n = idx % DECODE_N_TILE;

            int global_k = k0 + k;
            int global_col = blockIdx.x * DECODE_N_TILE + n;

            if (global_k < K && global_col < N) {
                sB[k][n] =
                    B[static_cast<size_t>(global_k) *
                      static_cast<size_t>(N) +
                      static_cast<size_t>(global_col)];
            } else {
                sB[k][n] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < DECODE_K_TILE; k++) {
            acc += sA[local_row][k] * sB[k][local_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        size_t c_idx =
            static_cast<size_t>(row) *
            static_cast<size_t>(N) +
            static_cast<size_t>(col);

        float old = beta == 0.0f ? 0.0f : C[c_idx];
        float b = bias == nullptr ? 0.0f : bias[col];

        float out = alpha * acc + beta * old + b;

        if (apply_gelu) {
            out = gelu_one(out);
        }

        C[c_idx] = out;
    }
}

void launch_decode_gemm_impl(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    const float* bias,
    float* C,
    cudaStream_t stream,
    bool apply_gelu
) {
    dim3 block(DECODE_N_TILE, DECODE_M_TILE);
    dim3 grid(
        (N + DECODE_N_TILE - 1) / DECODE_N_TILE,
        (M + DECODE_M_TILE - 1) / DECODE_M_TILE
    );

    decode_small_m_gemm_kernel<<<grid, block, 0, stream>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        bias,
        C,
        apply_gelu
    );
}

} // anonymous namespace

bool can_use_decode_gemm(
    int M,
    int N,
    int K
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return false;
    }

    /*
        Row-wise decode GEMM is useful for very small M.

        M = 1,2,4:
            row-wise [1,32] tile avoids wasted rows.

        M >= 8:
            row-wise blocks lose weight-tile reuse for qkv/fc1,
            so fall back to the general GEMM path.
    */
    if (M >= 4) {
        return false;
    }

    constexpr int D = mini_llm::constants::GPT2_D_MODEL;
    constexpr int FF = mini_llm::constants::GPT2_D_FF;

    bool is_qkv =
        (K == D && N == 3 * D);

    bool is_attn_out =
        (K == D && N == D);

    bool is_fc1 =
        (K == D && N == FF);

    bool is_fc2 =
        (K == FF && N == D);

    return is_qkv || is_attn_out || is_fc1 || is_fc2;
}

void launch_decode_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    const float* bias,
    float* C,
    cudaStream_t stream
) {
    launch_decode_gemm_impl(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        bias,
        C,
        stream,
        false
    );
}

void launch_decode_gemm_gelu(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    const float* bias,
    float* C,
    cudaStream_t stream
) {
    launch_decode_gemm_impl(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        bias,
        C,
        stream,
        true
    );
}

} // namespace mini_llm::kernels
