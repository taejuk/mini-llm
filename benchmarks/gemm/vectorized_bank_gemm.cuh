#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

// sA is stored as transposed layout: sA[k][row].
// If stride is BM=64, then stride % 32 == 0,
// so different k values can map to the same shared memory bank.
// Add padding to reduce bank conflicts while keeping float4 alignment.
#define SA_STRIDE (BM + 4)

#define NUM_THREADS ((BM / TM) * (BN / TN))

__global__ void sgemm_vectorized_bank(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    // sA layout:
    //   logical A tile: [BM x BK]
    //   shared layout: [BK x SA_STRIDE]
    //   sA[k][row]
    //
    // sB layout:
    //   logical B tile: [BK x BN]
    //   shared layout: [BK x BN]
    //   sB[k][col]
    __shared__ float sA[BK * SA_STRIDE];
    __shared__ float sB[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM][TN] = {0.0f};

    float regA[TM];
    float regB[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ------------------------------------------------------------
        // Load A tile from global memory to shared memory.
        //
        // Global A tile shape:
        //   A[BM x BK]
        //
        // Each row has BK=8 floats.
        // With float4, each row has 2 vector loads.
        //
        // Store transposed into shared memory:
        //   sA[k][row]
        //
        // Padding SA_STRIDE reduces bank conflict during this transpose store.
        // ------------------------------------------------------------
        for (int i = 0; i < (BM * BK) / (NUM_THREADS * 4); i++) {
            int pairIdx = threadIdx.x + i * NUM_THREADS;

            int rowA = pairIdx / (BK / 4);
            int colA4 = pairIdx % (BK / 4);

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &A[rowA * K + colA4 * 4]
                )[0];

            sA[(colA4 * 4 + 0) * SA_STRIDE + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * SA_STRIDE + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * SA_STRIDE + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * SA_STRIDE + rowA] = tmp.w;
        }

        // ------------------------------------------------------------
        // Load B tile from global memory to shared memory.
        //
        // Global B tile shape:
        //   B[BK x BN]
        //
        // B is row-major, so loading continuous columns using float4 is coalesced.
        // ------------------------------------------------------------
        for (int i = 0; i < (BK * BN) / (NUM_THREADS * 4); i++) {
            int pairIdx = threadIdx.x + i * NUM_THREADS;

            int rowB = pairIdx / (BN / 4);
            int colB4 = pairIdx % (BN / 4);

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &B[rowB * N + colB4 * 4]
                )[0];

            reinterpret_cast<float4*>(
                &sB[rowB * BN + colB4 * 4]
            )[0] = tmp;
        }

        __syncthreads();

        // Move A/B base pointers to next K tile.
        A += BK;
        B += BK * N;

        // ------------------------------------------------------------
        // Compute C[TM x TN] per thread.
        //
        // Each thread computes 8x8 output elements.
        // A values are reused across TN columns.
        // B values are reused across TM rows.
        // ------------------------------------------------------------
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            const int aOffset = dotIdx * SA_STRIDE + threadRow * TM;
            const int bOffset = dotIdx * BN + threadCol * TN;

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
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // ------------------------------------------------------------
    // Store C tile.
    //
    // Each thread stores 8x8 values.
    // Store uses float4.
    // ------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        float* c0_ptr =
            &C[(threadRow * TM + i) * N + threadCol * TN];

        float* c1_ptr =
            &C[(threadRow * TM + i) * N + threadCol * TN + 4];

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

void vectorized_gemm_bank(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    // This kernel is optimized for benchmark sizes that are multiples of
    // BM, BN, and BK.
    //
    // The original kernel also assumes aligned float4 loads/stores.
    // For general M/N/K, add boundary handling or use a fallback kernel.
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
        printf(
            "vectorized_gemm requires M %% %d == 0, N %% %d == 0, K %% %d == 0. "
            "Got M=%d, N=%d, K=%d\n",
            BM,
            BN,
            BK,
            M,
            N,
            K
        );
        return;
    }

    // blockIdx.x corresponds to column tile.
    // blockIdx.y corresponds to row tile.
    dim3 gridDim(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    dim3 blockDim(NUM_THREADS);

    sgemm_vectorized_bank<<<gridDim, blockDim>>>(
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