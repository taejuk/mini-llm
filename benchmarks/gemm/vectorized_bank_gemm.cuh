#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define VBANK_BM 64
#define VBANK_BN 64
#define VBANK_BK 8
#define VBANK_TM 8
#define VBANK_TN 8

#define VBANK_SA_STRIDE (VBANK_BM + 4)

#define VBANK_NUM_THREADS ((VBANK_BM / VBANK_TM) * (VBANK_BN / VBANK_TN))

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

    const int threadRow = threadIdx.x / (VBANK_BN / VBANK_TN);
    const int threadCol = threadIdx.x % (VBANK_BN / VBANK_TN);

    __shared__ float sA[VBANK_BK * VBANK_SA_STRIDE];
    __shared__ float sB[VBANK_BK * VBANK_BN];

    A += cRow * VBANK_BM * K;
    B += cCol * VBANK_BN;
    C += cRow * VBANK_BM * N + cCol * VBANK_BN;

    float threadResults[VBANK_TM][VBANK_TN] = {0.0f};

    float regA[VBANK_TM];
    float regB[VBANK_TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += VBANK_BK) {
        for (int i = 0; i < (VBANK_BM * VBANK_BK) / (VBANK_NUM_THREADS * 4); i++) {
            int pairIdx = threadIdx.x + i * VBANK_NUM_THREADS;

            int rowA = pairIdx / (VBANK_BK / 4);
            int colA4 = pairIdx % (VBANK_BK / 4);

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &A[rowA * K + colA4 * 4]
                )[0];

            sA[(colA4 * 4 + 0) * VBANK_SA_STRIDE + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * VBANK_SA_STRIDE + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * VBANK_SA_STRIDE + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * VBANK_SA_STRIDE + rowA] = tmp.w;
        }

        for (int i = 0; i < (VBANK_BK * VBANK_BN) / (VBANK_NUM_THREADS * 4); i++) {
            int pairIdx = threadIdx.x + i * VBANK_NUM_THREADS;

            int rowB = pairIdx / (VBANK_BN / 4);
            int colB4 = pairIdx % (VBANK_BN / 4);

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &B[rowB * N + colB4 * 4]
                )[0];

            reinterpret_cast<float4*>(
                &sB[rowB * VBANK_BN + colB4 * 4]
            )[0] = tmp;
        }

        __syncthreads();

        A += VBANK_BK;
        B += VBANK_BK * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < VBANK_BK; dotIdx++) {
            const int aOffset =
                dotIdx * VBANK_SA_STRIDE + threadRow * VBANK_TM;

            const int bOffset =
                dotIdx * VBANK_BN + threadCol * VBANK_TN;

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
            for (int i = 0; i < VBANK_TM; i++) {
                #pragma unroll
                for (int j = 0; j < VBANK_TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < VBANK_TM; i++) {
        float* c0_ptr =
            &C[(threadRow * VBANK_TM + i) * N + threadCol * VBANK_TN];

        float* c1_ptr =
            &C[(threadRow * VBANK_TM + i) * N + threadCol * VBANK_TN + 4];

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

void vectorized_bank_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    if ((M % VBANK_BM) != 0 ||
        (N % VBANK_BN) != 0 ||
        (K % VBANK_BK) != 0) {
        printf(
            "vectorized_bank_gemm requires M %% %d == 0, "
            "N %% %d == 0, K %% %d == 0. Got M=%d, N=%d, K=%d\n",
            VBANK_BM,
            VBANK_BN,
            VBANK_BK,
            M,
            N,
            K
        );
        return;
    }

    dim3 gridDim(
        (N + VBANK_BN - 1) / VBANK_BN,
        (M + VBANK_BM - 1) / VBANK_BM
    );

    dim3 blockDim(VBANK_NUM_THREADS);

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
