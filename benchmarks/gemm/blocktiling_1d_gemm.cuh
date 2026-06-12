#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BT1D_BM 64
#define BT1D_BN 64
#define BT1D_BK 8
#define BT1D_TM 8

__global__ void sgemm_blocktiling_1d(
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

    const int threadRow = threadIdx.x / BT1D_BN;
    const int threadCol = threadIdx.x % BT1D_BN;

    __shared__ float sA[BT1D_BM * BT1D_BK];
    __shared__ float sB[BT1D_BK * BT1D_BN];

    A += cRow * BT1D_BM * K;
    B += cCol * BT1D_BN;
    C += cRow * BT1D_BM * N + cCol * BT1D_BN;

    float threadResults[BT1D_TM] = {0.0f};

    const int innerRowA = threadIdx.x / BT1D_BK;
    const int innerColA = threadIdx.x % BT1D_BK;

    const int innerRowB = threadIdx.x / BT1D_BN;
    const int innerColB = threadIdx.x % BT1D_BN;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BT1D_BK) {
        sA[innerRowA * BT1D_BK + innerColA] =
            A[innerRowA * K + innerColA];

        sB[innerRowB * BT1D_BN + innerColB] =
            B[innerRowB * N + innerColB];

        __syncthreads();

        A += BT1D_BK;
        B += BT1D_BK * N;

        for (int dotIdx = 0; dotIdx < BT1D_BK; dotIdx++) {
            float tmpB = sB[dotIdx * BT1D_BN + threadCol];

            for (int resIdx = 0; resIdx < BT1D_TM; resIdx++) {
                threadResults[resIdx] +=
                    sA[(threadRow * BT1D_TM + resIdx) * BT1D_BK + dotIdx] *
                    tmpB;
            }
        }

        __syncthreads();
    }

    for (int resIdx = 0; resIdx < BT1D_TM; resIdx++) {
        C[(threadRow * BT1D_TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] +
            beta * C[(threadRow * BT1D_TM + resIdx) * N + threadCol];
    }
}

void blocktiling_1d_gemm(
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
        (N + BT1D_BN - 1) / BT1D_BN,
        (M + BT1D_BM - 1) / BT1D_BM
    );

    dim3 blockDim(BT1D_BM * BT1D_BN / BT1D_TM);

    sgemm_blocktiling_1d<<<gridDim, blockDim>>>(
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
