#pragma once
#include <cuda_runtime.h>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
namespace mini_llm::kernels {
// C = beta*C + alpha* (AxB);
__global__ void gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
);
}
