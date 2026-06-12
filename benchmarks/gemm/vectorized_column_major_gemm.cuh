#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

// -----------------------------------------------------------------------------
// vectorized_pretrans_a_gemm.cuh
//
// This file implements an SGEMM variant that assumes A is already stored in
// transposed global-memory layout.
//
// Original GEMM:
//   C[M x N] = A[M x K] x B[K x N]
//
// This kernel expects:
//   AT[K x M], where AT[k][m] = A[m][k]
//
// Why?
//   The original vectorized_gemm loads row-major A and stores it into shared
//   memory as transposed sA[k][row]. That transpose store can introduce shared
//   memory bank conflicts.
//
//   If A is pre-transposed into AT, then the global layout already matches the
//   desired shared-memory layout. We can do:
//
//      global AT[k][row : row+4]
//        -> float4 load
//        -> shared sA[k][row : row+4] with float4 store
//
// Usage:
//   1. Allocate dAT with size K * M.
//   2. Call transpose_A_for_vectorized_pretrans_a(M, K, dA, dAT) once.
//   3. Benchmark vectorized_pretrans_a_gemm(..., dAT, dB, ..., dC).
//
// Important:
//   The first matrix argument of vectorized_pretrans_a_gemm is AT, not A.
// -----------------------------------------------------------------------------

#define PAT_BM 64
#define PAT_BN 64
#define PAT_BK 8
#define PAT_TM 8
#define PAT_TN 8

// sA is stored as sA[k][row].
// Keep +4 padding to preserve float4 alignment and reduce k-stride bank aliasing.
#define PAT_SA_STRIDE (PAT_BM + 4)

#define PAT_NUM_THREADS ((PAT_BM / PAT_TM) * (PAT_BN / PAT_TN))

#define PAT_TRANSPOSE_TILE_DIM 32
#define PAT_TRANSPOSE_BLOCK_ROWS 8

// -----------------------------------------------------------------------------
// Tiled transpose kernel
//
// Input:
//   A  [M x K], row-major
//
// Output:
//   AT [K x M], row-major
//   AT[k * M + m] = A[m * K + k]
//
// The 32x33 shared tile avoids bank conflicts in the transpose kernel itself.
// -----------------------------------------------------------------------------
__global__ void transpose_A_tiled_kernel(
    int M,
    int K,
    const float* __restrict__ A,
    float* __restrict__ AT
) {
    __shared__ float tile[PAT_TRANSPOSE_TILE_DIM][PAT_TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * PAT_TRANSPOSE_TILE_DIM + threadIdx.x; // k index
    int y = blockIdx.y * PAT_TRANSPOSE_TILE_DIM + threadIdx.y; // m index

    #pragma unroll
    for (int j = 0; j < PAT_TRANSPOSE_TILE_DIM; j += PAT_TRANSPOSE_BLOCK_ROWS) {
        if (x < K && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] =
                A[static_cast<size_t>(y + j) * K + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * PAT_TRANSPOSE_TILE_DIM + threadIdx.x; // m index
    y = blockIdx.x * PAT_TRANSPOSE_TILE_DIM + threadIdx.y; // k index

    #pragma unroll
    for (int j = 0; j < PAT_TRANSPOSE_TILE_DIM; j += PAT_TRANSPOSE_BLOCK_ROWS) {
        if (x < M && (y + j) < K) {
            AT[static_cast<size_t>(y + j) * M + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void transpose_A_for_vectorized_column_major_a(
    int M,
    int K,
    const float* A,
    float* AT
) {
    dim3 block(PAT_TRANSPOSE_TILE_DIM, PAT_TRANSPOSE_BLOCK_ROWS);
    dim3 grid(
        (K + PAT_TRANSPOSE_TILE_DIM - 1) / PAT_TRANSPOSE_TILE_DIM,
        (M + PAT_TRANSPOSE_TILE_DIM - 1) / PAT_TRANSPOSE_TILE_DIM
    );

    transpose_A_tiled_kernel<<<grid, block>>>(M, K, A, AT);
}

// -----------------------------------------------------------------------------
// GEMM kernel using pre-transposed A.
//
// AT layout:
//   AT [K x M]
//   AT[k][m] = A[m][k]
//
// B layout:
//   B [K x N], row-major
//
// C layout:
//   C [M x N], row-major
// -----------------------------------------------------------------------------
__global__ void vectorized_column_major_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ AT,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (PAT_BN / PAT_TN);
    const int threadCol = threadIdx.x % (PAT_BN / PAT_TN);

    // sA layout:
    //   logical A tile: [PAT_BM x PAT_BK]
    //   shared layout: [PAT_BK x PAT_SA_STRIDE]
    //   sA[k][row]
    //
    // Since AT is already [K x M], loading AT[k][row:row+4] and storing
    // directly to sA[k][row:row+4] removes the scalar transpose store.
    __shared__ float sA[PAT_BK * PAT_SA_STRIDE];
    __shared__ float sB[PAT_BK * PAT_BN];

    C += static_cast<size_t>(cRow) * PAT_BM * N + cCol * PAT_BN;

    float threadResults[PAT_TM][PAT_TN] = {0.0f};

    float regA[PAT_TM];
    float regB[PAT_TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += PAT_BK) {
        // ---------------------------------------------------------------------
        // Load pre-transposed A tile:
        //
        // AT tile shape for current CTA:
        //   AT[bkIdx : bkIdx+PAT_BK][cRow*PAT_BM : cRow*PAT_BM+PAT_BM]
        //
        // This is exactly the sA[k][row] layout.
        // ---------------------------------------------------------------------
        constexpr int A_VEC_PER_K = PAT_BM / 4;
        constexpr int A_VEC_TOTAL = PAT_BK * A_VEC_PER_K;

        for (int i = 0; i < A_VEC_TOTAL / PAT_NUM_THREADS; i++) {
            int vecIdx = threadIdx.x + i * PAT_NUM_THREADS;

            int localK = vecIdx / A_VEC_PER_K;
            int rowA4 = vecIdx % A_VEC_PER_K;

            int globalK = bkIdx + localK;
            int globalRow = cRow * PAT_BM + rowA4 * 4;

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &AT[static_cast<size_t>(globalK) * M + globalRow]
                )[0];

            reinterpret_cast<float4*>(
                &sA[localK * PAT_SA_STRIDE + rowA4 * 4]
            )[0] = tmp;
        }

        // ---------------------------------------------------------------------
        // Load B tile:
        //
        // B tile shape:
        //   B[bkIdx : bkIdx+PAT_BK][cCol*PAT_BN : cCol*PAT_BN+PAT_BN]
        //
        // B is row-major, so this is coalesced float4 loading.
        // ---------------------------------------------------------------------
        constexpr int B_VEC_PER_ROW = PAT_BN / 4;
        constexpr int B_VEC_TOTAL = PAT_BK * B_VEC_PER_ROW;

        for (int i = 0; i < B_VEC_TOTAL / PAT_NUM_THREADS; i++) {
            int vecIdx = threadIdx.x + i * PAT_NUM_THREADS;

            int rowB = vecIdx / B_VEC_PER_ROW;
            int colB4 = vecIdx % B_VEC_PER_ROW;

            int globalK = bkIdx + rowB;
            int globalCol = cCol * PAT_BN + colB4 * 4;

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &B[static_cast<size_t>(globalK) * N + globalCol]
                )[0];

            reinterpret_cast<float4*>(
                &sB[rowB * PAT_BN + colB4 * 4]
            )[0] = tmp;
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // Compute C[8 x 8] per thread.
        // ---------------------------------------------------------------------
        #pragma unroll
        for (int dotIdx = 0; dotIdx < PAT_BK; dotIdx++) {
            const int aOffset = dotIdx * PAT_SA_STRIDE + threadRow * PAT_TM;
            const int bOffset = dotIdx * PAT_BN + threadCol * PAT_TN;

            const float4 a0 =
                reinterpret_cast<float4*>(&sA[aOffset])[0];

            const float4 a1 =
                reinterpret_cast<float4*>(&sA[aOffset + 4])[0];

            regA[0] = a0.x;
            regA[1] = a0.y;
            regA[2] = a0.z;
            regA[3] = a0.w;

            regA[4] = a1.x;
            regA[5] = a1.y;
            regA[6] = a1.z;
            regA[7] = a1.w;

            const float4 b0 =
                reinterpret_cast<float4*>(&sB[bOffset])[0];

            const float4 b1 =
                reinterpret_cast<float4*>(&sB[bOffset + 4])[0];

            regB[0] = b0.x;
            regB[1] = b0.y;
            regB[2] = b0.z;
            regB[3] = b0.w;

            regB[4] = b1.x;
            regB[5] = b1.y;
            regB[6] = b1.z;
            regB[7] = b1.w;

            #pragma unroll
            for (int i = 0; i < PAT_TM; i++) {
                #pragma unroll
                for (int j = 0; j < PAT_TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Store C tile.
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < PAT_TM; i++) {
        float* c0_ptr =
            &C[(threadRow * PAT_TM + i) * N + threadCol * PAT_TN];

        float* c1_ptr =
            &C[(threadRow * PAT_TM + i) * N + threadCol * PAT_TN + 4];

        if (beta == 0.0f) {
            reinterpret_cast<float4*>(c0_ptr)[0] = make_float4(
                alpha * threadResults[i][0],
                alpha * threadResults[i][1],
                alpha * threadResults[i][2],
                alpha * threadResults[i][3]
            );

            reinterpret_cast<float4*>(c1_ptr)[0] = make_float4(
                alpha * threadResults[i][4],
                alpha * threadResults[i][5],
                alpha * threadResults[i][6],
                alpha * threadResults[i][7]
            );
        } else {
            float4 ec0 = reinterpret_cast<float4*>(c0_ptr)[0];
            float4 ec1 = reinterpret_cast<float4*>(c1_ptr)[0];

            reinterpret_cast<float4*>(c0_ptr)[0] = make_float4(
                alpha * threadResults[i][0] + beta * ec0.x,
                alpha * threadResults[i][1] + beta * ec0.y,
                alpha * threadResults[i][2] + beta * ec0.z,
                alpha * threadResults[i][3] + beta * ec0.w
            );

            reinterpret_cast<float4*>(c1_ptr)[0] = make_float4(
                alpha * threadResults[i][4] + beta * ec1.x,
                alpha * threadResults[i][5] + beta * ec1.y,
                alpha * threadResults[i][6] + beta * ec1.z,
                alpha * threadResults[i][7] + beta * ec1.w
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Wrapper for benchmark.
//
// IMPORTANT:
//   The first matrix argument is AT, not A.
//
//   vectorized_pretrans_a_gemm(M, N, K, alpha, dAT, dB, beta, dC)
// -----------------------------------------------------------------------------
void vectorized_column_major_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* AT,
    const float* B,
    float beta,
    float* C
) {
    if ((M % PAT_BM) != 0 || (N % PAT_BN) != 0 || (K % PAT_BK) != 0) {
        printf(
            "vectorized_pretrans_a_gemm requires M %% %d == 0, "
            "N %% %d == 0, K %% %d == 0. Got M=%d, N=%d, K=%d\n",
            PAT_BM,
            PAT_BN,
            PAT_BK,
            M,
            N,
            K
        );
        return;
    }

    dim3 gridDim(
        (N + PAT_BN - 1) / PAT_BN,
        (M + PAT_BM - 1) / PAT_BM
    );

    dim3 blockDim(PAT_NUM_THREADS);

    vectorized_column_major_gemm_kernel<<<gridDim, blockDim>>>(
        M,
        N,
        K,
        alpha,
        AT,
        B,
        beta,
        C
    );
}
