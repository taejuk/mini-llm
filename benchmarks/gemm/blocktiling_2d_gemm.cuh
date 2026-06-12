#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define BT2D_BM 64
#define BT2D_BN 64
#define BT2D_BK 8
#define BT2D_TM 8
#define BT2D_TN 8

#define BT2D_NUM_THREADS ((BT2D_BM / BT2D_TM) * (BT2D_BN / BT2D_TN))

__global__ void sgemm_blocktiling_2d(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BT2D_BN / BT2D_TN);
    const int threadCol = threadIdx.x % (BT2D_BN / BT2D_TN);

    __shared__ float sA[BT2D_BM * BT2D_BK];
    __shared__ float sB[BT2D_BK * BT2D_BN];

    A += cRow * BT2D_BM * K;
    B += cCol * BT2D_BN;
    C += cRow * BT2D_BM * N + cCol * BT2D_BN;

    float threadResults[BT2D_TM][BT2D_TN] = {0.0f};

    float regA[BT2D_TM];
    float regB[BT2D_TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BT2D_BK) {
        for (int i = 0; i < (BT2D_BM * BT2D_BK) / BT2D_NUM_THREADS; i++) {
            int idx = threadIdx.x + i * BT2D_NUM_THREADS;
            int rowA = idx / BT2D_BK;
            int colA = idx % BT2D_BK;

            sA[idx] = A[rowA * K + colA];
        }

        for (int i = 0; i < (BT2D_BK * BT2D_BN) / BT2D_NUM_THREADS; i++) {
            int idx = threadIdx.x + i * BT2D_NUM_THREADS;
            int rowB = idx / BT2D_BN;
            int colB = idx % BT2D_BN;

            sB[idx] = B[rowB * N + colB];
        }

        __syncthreads();

        A += BT2D_BK;
        B += BT2D_BK * N;

        for (int dotIdx = 0; dotIdx < BT2D_BK; dotIdx++) {
            for (int i = 0; i < BT2D_TM; i++) {
                regA[i] =
                    sA[(threadRow * BT2D_TM + i) * BT2D_BK + dotIdx];
            }

            for (int j = 0; j < BT2D_TN; j++) {
                regB[j] =
                    sB[dotIdx * BT2D_BN + threadCol * BT2D_TN + j];
            }

            for (int i = 0; i < BT2D_TM; i++) {
                for (int j = 0; j < BT2D_TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < BT2D_TM; i++) {
        for (int j = 0; j < BT2D_TN; j++) {
            C[(threadRow * BT2D_TM + i) * N + threadCol * BT2D_TN + j] =
                alpha * threadResults[i][j] +
                beta * C[(threadRow * BT2D_TM + i) * N +
                         threadCol * BT2D_TN + j];
        }
    }
}

void blocktiling_2d_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    dim3 gridDim(
        (N + BT2D_BN - 1) / BT2D_BN,
        (M + BT2D_BM - 1) / BT2D_BM
    );

    dim3 blockDim(BT2D_NUM_THREADS);

    sgemm_blocktiling_2d<<<gridDim, blockDim>>>(
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
