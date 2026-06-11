#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

__global__ void wmma_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const half* __restrict__ A,
    const half* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    int warpRow = blockIdx.y;
    int warpCol = blockIdx.x;

    int row = warpRow * WMMA_TILE_M;
    int col = warpCol * WMMA_TILE_N;

    if (row >= M || col >= N) {
        return;
    }

    wmma::fragment<
        wmma::accumulator,
        WMMA_TILE_M,
        WMMA_TILE_N,
        WMMA_TILE_K,
        float
    > frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);

    for (int k_step = 0; k_step < K; k_step += WMMA_TILE_K) {
        wmma::fragment<
            wmma::matrix_a,
            WMMA_TILE_M,
            WMMA_TILE_N,
            WMMA_TILE_K,
            half,
            wmma::row_major
        > frag_a;

        wmma::fragment<
            wmma::matrix_b,
            WMMA_TILE_M,
            WMMA_TILE_N,
            WMMA_TILE_K,
            half,
            wmma::row_major
        > frag_b;

        const half* a_tile_ptr =
            A + static_cast<size_t>(row) * K + k_step;

        const half* b_tile_ptr =
            B + static_cast<size_t>(k_step) * N + col;

        wmma::load_matrix_sync(frag_a, a_tile_ptr, K);
        wmma::load_matrix_sync(frag_b, b_tile_ptr, N);

        wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    }

    float* c_tile_ptr =
        C + static_cast<size_t>(row) * N + col;

    // Fast path: benchmark에서 주로 사용하는 C = A * B
    if (alpha == 1.0f && beta == 0.0f) {
        wmma::store_matrix_sync(
            c_tile_ptr,
            frag_acc,
            N,
            wmma::mem_row_major
        );
        return;
    }

    // General path: C = alpha * A * B + beta * C
    wmma::fragment<
        wmma::accumulator,
        WMMA_TILE_M,
        WMMA_TILE_N,
        WMMA_TILE_K,
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
    for (int i = 0; i < frag_acc.num_elements; i++) {
        frag_acc.x[i] = alpha * frag_acc.x[i] + beta * frag_old.x[i];
    }

    wmma::store_matrix_sync(
        c_tile_ptr,
        frag_acc,
        N,
        wmma::mem_row_major
    );
}

void wmma_gemm(
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

    if ((M % WMMA_TILE_M) != 0 ||
        (N % WMMA_TILE_N) != 0 ||
        (K % WMMA_TILE_K) != 0) {
        printf(
            "wmma_gemm requires M, N, K to be multiples of 16. "
            "Got M=%d, N=%d, K=%d\n",
            M,
            N,
            K
        );
        return;
    }

    dim3 block(32, 1, 1);

    dim3 grid(
        N / WMMA_TILE_N,
        M / WMMA_TILE_M,
        1
    );

    wmma_gemm_kernel<<<grid, block>>>(
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