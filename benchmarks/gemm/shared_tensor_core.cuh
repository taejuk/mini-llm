#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define STC_WMMA_M 16
#define STC_WMMA_N 16
#define STC_WMMA_K 16

#define STC_CTA_M 64
#define STC_CTA_N 64
#define STC_CTA_K 16

#define STC_WARPS_M 4
#define STC_WARPS_N 4
#define STC_WARPS_PER_BLOCK (STC_WARPS_M * STC_WARPS_N)
#define STC_THREADS_PER_BLOCK (STC_WARPS_PER_BLOCK * 32)

__global__ void shared_tensor_core_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const half* __restrict__ A,
    const half* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    int tid = threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_m = warp_id / STC_WARPS_N;
    int warp_n = warp_id % STC_WARPS_N;

    int block_row = blockIdx.y * STC_CTA_M;
    int block_col = blockIdx.x * STC_CTA_N;

    int warp_row = block_row + warp_m * STC_WMMA_M;
    int warp_col = block_col + warp_n * STC_WMMA_N;

    __shared__ __align__(16) half sA[STC_CTA_M * STC_CTA_K];
    __shared__ __align__(16) half sB[STC_CTA_K * STC_CTA_N];

    wmma::fragment<
        wmma::accumulator,
        STC_WMMA_M,
        STC_WMMA_N,
        STC_WMMA_K,
        float
    > frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    for (int k0 = 0; k0 < K; k0 += STC_CTA_K) {
        // ------------------------------------------------------------
        // Load A tile to shared memory
        // A tile shape: [64 x 16]
        // ------------------------------------------------------------
        for (
            int idx = tid;
            idx < STC_CTA_M * STC_CTA_K;
            idx += STC_THREADS_PER_BLOCK
        ) {
            int local_row = idx / STC_CTA_K;
            int local_k   = idx % STC_CTA_K;

            int global_row = block_row + local_row;
            int global_k   = k0 + local_k;

            half value = __float2half(0.0f);

            if (global_row < M && global_k < K) {
                value = A[
                    static_cast<size_t>(global_row) * K +
                    global_k
                ];
            }

            sA[local_row * STC_CTA_K + local_k] = value;
        }

        // ------------------------------------------------------------
        // Load B tile to shared memory
        // B tile shape: [16 x 64]
        // ------------------------------------------------------------
        for (
            int idx = tid;
            idx < STC_CTA_K * STC_CTA_N;
            idx += STC_THREADS_PER_BLOCK
        ) {
            int local_k   = idx / STC_CTA_N;
            int local_col = idx % STC_CTA_N;

            int global_k   = k0 + local_k;
            int global_col = block_col + local_col;

            half value = __float2half(0.0f);

            if (global_k < K && global_col < N) {
                value = B[
                    static_cast<size_t>(global_k) * N +
                    global_col
                ];
            }

            sB[local_k * STC_CTA_N + local_col] = value;
        }

        __syncthreads();

        // ------------------------------------------------------------
        // Each warp computes one C[16 x 16] tile
        // ------------------------------------------------------------
        wmma::fragment<
            wmma::matrix_a,
            STC_WMMA_M,
            STC_WMMA_N,
            STC_WMMA_K,
            half,
            wmma::row_major
        > frag_a;

        wmma::fragment<
            wmma::matrix_b,
            STC_WMMA_M,
            STC_WMMA_N,
            STC_WMMA_K,
            half,
            wmma::row_major
        > frag_b;

        const half* a_frag_ptr =
            sA + warp_m * STC_WMMA_M * STC_CTA_K;

        const half* b_frag_ptr =
            sB + warp_n * STC_WMMA_N;

        wmma::load_matrix_sync(
            frag_a,
            a_frag_ptr,
            STC_CTA_K
        );

        wmma::load_matrix_sync(
            frag_b,
            b_frag_ptr,
            STC_CTA_N
        );

        wmma::mma_sync(
            frag_c,
            frag_a,
            frag_b,
            frag_c
        );

        __syncthreads();
    }

    bool tile_valid =
        (warp_row + STC_WMMA_M <= M) &&
        (warp_col + STC_WMMA_N <= N);

    if (!tile_valid) {
        return;
    }

    float* c_tile_ptr =
        C + static_cast<size_t>(warp_row) * N + warp_col;

    // Fast path: C = A * B
    if (alpha == 1.0f && beta == 0.0f) {
        wmma::store_matrix_sync(
            c_tile_ptr,
            frag_c,
            N,
            wmma::mem_row_major
        );

        return;
    }

    // General path: C = alpha * A * B + beta * C
    wmma::fragment<
        wmma::accumulator,
        STC_WMMA_M,
        STC_WMMA_N,
        STC_WMMA_K,
        float
    > frag_old;

    if (beta != 0.0f) {
        wmma::load_matrix_sync(
            frag_old,
            c_tile_ptr,
            N,
            wmma::mem_row_major
        );
    } else {
        wmma::fill_fragment(frag_old, 0.0f);
    }

    #pragma unroll
    for (int i = 0; i < frag_c.num_elements; i++) {
        frag_c.x[i] = alpha * frag_c.x[i] + beta * frag_old.x[i];
    }

    wmma::store_matrix_sync(
        c_tile_ptr,
        frag_c,
        N,
        wmma::mem_row_major
    );
}

void shared_tensor_core_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if ((M % STC_WMMA_M) != 0 ||
        (N % STC_WMMA_N) != 0 ||
        (K % STC_WMMA_K) != 0) {
        printf(
            "shared_tensor_core_gemm requires M, N, K to be multiples of 16. "
            "Got M=%d, N=%d, K=%d\n",
            M,
            N,
            K
        );
        return;
    }

    dim3 block(STC_THREADS_PER_BLOCK);

    dim3 grid(
        (N + STC_CTA_N - 1) / STC_CTA_N,
        (M + STC_CTA_M - 1) / STC_CTA_M
    );

    shared_tensor_core_gemm_kernel<<<grid, block>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        C
    );
}
