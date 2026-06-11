#pragma once

#include <cuda_runtime.h>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

namespace mini_llm::kernels {

// C = beta * C + alpha * (A x B)
void launch_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    cudaStream_t stream = nullptr
);

// C = A x B + bias
void launch_gemm_bias(
    int M,
    int N,
    int K,
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    cudaStream_t stream = nullptr
);

} // namespace mini_llm::kernels
