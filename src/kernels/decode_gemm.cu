#include "kernels/decode_gemm.cuh"

#include <cuda_runtime.h>

namespace mini_llm::kernels {

namespace {

constexpr int DECODE_M_TILE = 8;
constexpr int DECODE_N_TILE = 32;
constexpr int DECODE_K_TILE = 64;

/*
    Decode small-M GEMM:

        C[M, N] = alpha * A[M, K] x B[K, N] + beta * C[M, N] + bias[N]

    Target shapes:
        qkv_projection      : M <= 16, K = 768,  N = 2304
        attn_out_projection : M <= 16, K = 768,  N = 768
        fc1                 : M <= 16, K = 768,  N = 3072
        fc2                 : M <= 16, K = 3072, N = 768
        vocab_gemm          : M <= 16, K = 768,  N = 50257

    One CUDA block computes:
        C tile [8 rows, 32 cols]

    Thread layout:
        blockDim.x = 32  -> output column
        blockDim.y = 8   -> batch row
*/
__global__ void decode_small_m_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    const float* __restrict__ bias,
    float* __restrict__ C
) {
    int local_col = threadIdx.x; // 0..31
    int local_row = threadIdx.y; // 0..7

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
                    A[static_cast<size_t>(global_row) * K + global_k];
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
                    B[static_cast<size_t>(global_k) * N + global_col];
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
            static_cast<size_t>(row) * static_cast<size_t>(N) +
            static_cast<size_t>(col);

        float old = beta == 0.0f ? 0.0f : C[c_idx];
        float b = bias == nullptr ? 0.0f : bias[col];

        C[c_idx] = alpha * acc + beta * old + b;
    }
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

    // Decode에서는 M=batch_size가 보통 1,2,4,8,16.
    // M이 커지면 기존 prefill용 vbank/general GEMM 쪽이 더 적합함.
    if (M > 16) {
        return false;
    }

    return true;
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
        C
    );
}

} // namespace mini_llm::kernels