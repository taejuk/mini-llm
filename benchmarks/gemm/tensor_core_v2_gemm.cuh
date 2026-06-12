#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ------------------------------------------------------------
// Tensor Core GEMM v2
//
// C = alpha * A x B + beta * C
//
// A: half, row-major [M x K]
// B: half, row-major [K x N]
// C: float, row-major [M x N]
//
// One CTA computes C[64 x 64]
// One warp computes C[16 x 16]
// One MMA computes 16x16x16
//
// Requirements:
//   M % 64 == 0
//   N % 64 == 0
//   K % 16 == 0
// ------------------------------------------------------------

#define TCV2_WMMA_M 16
#define TCV2_WMMA_N 16
#define TCV2_WMMA_K 16

#define TCV2_CTA_M 64
#define TCV2_CTA_N 64
#define TCV2_CTA_K 16

#define TCV2_WARPS_M 4
#define TCV2_WARPS_N 4
#define TCV2_WARPS_PER_BLOCK (TCV2_WARPS_M * TCV2_WARPS_N)
#define TCV2_THREADS_PER_BLOCK (TCV2_WARPS_PER_BLOCK * 32)

#define TCV2_HALF_PER_UINT4 8

union __align__(16) Half8Pack {
    uint4 u;
    half h[8];
};

__global__ void tensor_core_v2_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const half* __restrict__ A,
    const half* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int warp_m = warp_id / TCV2_WARPS_N;
    const int warp_n = warp_id % TCV2_WARPS_N;

    const int block_row = blockIdx.y * TCV2_CTA_M;
    const int block_col = blockIdx.x * TCV2_CTA_N;

    const int warp_row = block_row + warp_m * TCV2_WMMA_M;
    const int warp_col = block_col + warp_n * TCV2_WMMA_N;

    // ------------------------------------------------------------
    // Shared memory layout
    //
    // sA:
    //   A tile shape: [64 x 16]
    //   stored row-major
    //   sA[row][k]
    //
    // sB:
    //   B tile shape: [16 x 64]
    //   global B is row-major, but shared B is stored as col-major
    //   sB[col][k]
    //
    // Reason:
    //   - global B load remains coalesced
    //   - WMMA matrix_b uses col_major from shared memory
    // ------------------------------------------------------------
    __shared__ __align__(16) half sA[TCV2_CTA_M * TCV2_CTA_K];
    __shared__ __align__(16) half sB[TCV2_CTA_N * TCV2_CTA_K];

    wmma::fragment<
        wmma::accumulator,
        TCV2_WMMA_M,
        TCV2_WMMA_N,
        TCV2_WMMA_K,
        float
    > frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    // ------------------------------------------------------------
    // K loop
    //
    // Each step:
    //   C[64x64] += A[64x16] x B[16x64]
    // ------------------------------------------------------------
    for (int k0 = 0; k0 < K; k0 += TCV2_CTA_K) {
        // --------------------------------------------------------
        // Load A tile with vectorized coalesced load
        //
        // A tile: 64 x 16 half
        // 16 half = 32 bytes per row = 2 x uint4
        // Total uint4 loads = 64 * 2 = 128
        // --------------------------------------------------------
        constexpr int A_VEC_PER_ROW = TCV2_CTA_K / TCV2_HALF_PER_UINT4;
        constexpr int A_VEC_TOTAL = TCV2_CTA_M * A_VEC_PER_ROW;

        for (int vec_idx = tid; vec_idx < A_VEC_TOTAL; vec_idx += TCV2_THREADS_PER_BLOCK) {
            const int local_row = vec_idx / A_VEC_PER_ROW;
            const int local_vec = vec_idx % A_VEC_PER_ROW;

            const int global_row = block_row + local_row;
            const int global_k = k0 + local_vec * TCV2_HALF_PER_UINT4;

            const half* g_ptr =
                A + static_cast<size_t>(global_row) * K + global_k;

            half* s_ptr =
                sA + local_row * TCV2_CTA_K + local_vec * TCV2_HALF_PER_UINT4;

            uint4 v = *reinterpret_cast<const uint4*>(g_ptr);
            *reinterpret_cast<uint4*>(s_ptr) = v;
        }

        // --------------------------------------------------------
        // Load B tile with vectorized coalesced load
        //
        // Global B tile: [16 x 64], row-major
        // Each thread reads continuous B[k][col:col+8].
        //
        // Then store to shared memory as col-major:
        //   sB[col][k]
        // --------------------------------------------------------
        constexpr int B_VEC_PER_ROW = TCV2_CTA_N / TCV2_HALF_PER_UINT4;
        constexpr int B_VEC_TOTAL = TCV2_CTA_K * B_VEC_PER_ROW;

        for (int vec_idx = tid; vec_idx < B_VEC_TOTAL; vec_idx += TCV2_THREADS_PER_BLOCK) {
            const int local_k = vec_idx / B_VEC_PER_ROW;
            const int local_vec = vec_idx % B_VEC_PER_ROW;

            const int global_k = k0 + local_k;
            const int global_col = block_col + local_vec * TCV2_HALF_PER_UINT4;

            const half* g_ptr =
                B + static_cast<size_t>(global_k) * N + global_col;

            Half8Pack pack;
            pack.u = *reinterpret_cast<const uint4*>(g_ptr);

            #pragma unroll
            for (int i = 0; i < TCV2_HALF_PER_UINT4; i++) {
                const int local_col = local_vec * TCV2_HALF_PER_UINT4 + i;

                // sB is logically [CTA_N x CTA_K]
                // col-major view for WMMA matrix_b
                sB[local_col * TCV2_CTA_K + local_k] = pack.h[i];
            }
        }

        __syncthreads();

        // --------------------------------------------------------
        // WMMA load from shared memory
        //
        // Each warp computes one 16x16 tile of C.
        // --------------------------------------------------------
        wmma::fragment<
            wmma::matrix_a,
            TCV2_WMMA_M,
            TCV2_WMMA_N,
            TCV2_WMMA_K,
            half,
            wmma::row_major
        > frag_a;

        wmma::fragment<
            wmma::matrix_b,
            TCV2_WMMA_M,
            TCV2_WMMA_N,
            TCV2_WMMA_K,
            half,
            wmma::col_major
        > frag_b;

        const half* a_tile_ptr =
            sA + warp_m * TCV2_WMMA_M * TCV2_CTA_K;

        const half* b_tile_ptr =
            sB + warp_n * TCV2_WMMA_N * TCV2_CTA_K;

        wmma::load_matrix_sync(
            frag_a,
            a_tile_ptr,
            TCV2_CTA_K
        );

        wmma::load_matrix_sync(
            frag_b,
            b_tile_ptr,
            TCV2_CTA_K
        );

        // Accumulator fragment is reused across all K steps.
        wmma::mma_sync(
            frag_c,
            frag_a,
            frag_b,
            frag_c
        );

        __syncthreads();
    }

    // ------------------------------------------------------------
    // Store C
    // ------------------------------------------------------------
    float* c_tile_ptr =
        C + static_cast<size_t>(warp_row) * N + warp_col;

    if (alpha == 1.0f && beta == 0.0f) {
        wmma::store_matrix_sync(
            c_tile_ptr,
            frag_c,
            N,
            wmma::mem_row_major
        );

        return;
    }

    wmma::fragment<
        wmma::accumulator,
        TCV2_WMMA_M,
        TCV2_WMMA_N,
        TCV2_WMMA_K,
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

void tensor_core_v2_gemm(
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

    if ((M % TCV2_CTA_M) != 0 ||
        (N % TCV2_CTA_N) != 0 ||
        (K % TCV2_CTA_K) != 0) {
        printf(
            "tensor_core_v2_gemm requires M %% 64 == 0, "
            "N %% 64 == 0, K %% 16 == 0. Got M=%d, N=%d, K=%d\n",
            M,
            N,
            K
        );
        return;
    }

    dim3 block(TCV2_THREADS_PER_BLOCK);

    dim3 grid(
        N / TCV2_CTA_N,
        M / TCV2_CTA_M
    );

    tensor_core_v2_gemm_kernel<<<grid, block>>>(
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