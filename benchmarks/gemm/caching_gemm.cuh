#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CACHE_BLOCK_SIZE 16

__global__ void sgemm_caching(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    __shared__ float sA[CACHE_BLOCK_SIZE * CACHE_BLOCK_SIZE];
    __shared__ float sB[CACHE_BLOCK_SIZE * CACHE_BLOCK_SIZE];

    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    const int aOffset = cRow * K * CACHE_BLOCK_SIZE;
    const int bOffset = cCol * CACHE_BLOCK_SIZE;

    const int threadx = threadIdx.x;
    const int thready = threadIdx.y;

    float tmp = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += CACHE_BLOCK_SIZE) {
        sA[thready * CACHE_BLOCK_SIZE + threadx] =
            A[aOffset + K * threadx + bkIdx + thready];

        sB[threadx * CACHE_BLOCK_SIZE + thready] =
            B[bOffset + N * (bkIdx + threadx) + thready];

        __syncthreads();

        for (int dotIdx = 0; dotIdx < CACHE_BLOCK_SIZE; dotIdx++) {
            tmp +=
                sA[dotIdx * CACHE_BLOCK_SIZE + threadx] *
                sB[dotIdx * CACHE_BLOCK_SIZE + thready];
        }

        __syncthreads();
    }

    C[(cRow * CACHE_BLOCK_SIZE + threadx) * N +
      cCol * CACHE_BLOCK_SIZE + thready] =
        alpha * tmp +
        beta * C[(cRow * CACHE_BLOCK_SIZE + threadx) * N +
                 cCol * CACHE_BLOCK_SIZE + thready];
}

void caching_gemm(
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
        (M + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE,
        (N + CACHE_BLOCK_SIZE - 1) / CACHE_BLOCK_SIZE
    );

    dim3 blockDim(
        CACHE_BLOCK_SIZE,
        CACHE_BLOCK_SIZE
    );

    sgemm_caching<<<gridDim, blockDim>>>(
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
