#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define NATIVE_BLOCK_SIZE 32

__global__ void sgemm_naive(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void naive_gemm(
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
        (M + NATIVE_BLOCK_SIZE - 1) / NATIVE_BLOCK_SIZE,
        (N + NATIVE_BLOCK_SIZE - 1) / NATIVE_BLOCK_SIZE,
        1
    );

    dim3 blockDim(
        NATIVE_BLOCK_SIZE,
        NATIVE_BLOCK_SIZE,
        1
    );

    sgemm_naive<<<gridDim, blockDim>>>(
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
