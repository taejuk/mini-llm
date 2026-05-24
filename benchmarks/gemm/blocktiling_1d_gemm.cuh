#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void sgemm_blocktiling_1d(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / BN; 
    const int threadCol = threadIdx.x % BN;

    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM] = {0.0f};

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;

    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
        sA[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        sB[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float tmpB = sB[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                threadResults[resIdx] +=
                    sA[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; resIdx++) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

void blocktiling_1d_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
  dim3 blockDim(BM * BN / TM);
  sgemm_blocktiling_1d<<<gridDim, blockDim>>>(M, N, K, alpha ,A, B, beta,C);
}
