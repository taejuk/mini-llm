#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define COAL_BLOCK_SIZE 32

__global__ void sgemm_coalescing(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    const int x =
        blockIdx.x * COAL_BLOCK_SIZE +
        (threadIdx.x / COAL_BLOCK_SIZE);

    const int y =
        blockIdx.y * COAL_BLOCK_SIZE +
        (threadIdx.x % COAL_BLOCK_SIZE);

    if (x < M && y < N) {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void coalescing_gemm(
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
        (M + COAL_BLOCK_SIZE - 1) / COAL_BLOCK_SIZE,
        (N + COAL_BLOCK_SIZE - 1) / COAL_BLOCK_SIZE
    );

    dim3 blockDim(COAL_BLOCK_SIZE * COAL_BLOCK_SIZE);

    sgemm_coalescing<<<gridDim, blockDim>>>(
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
