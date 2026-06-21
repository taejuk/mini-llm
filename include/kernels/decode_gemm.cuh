#pragma once

#include <cuda_runtime.h>

namespace mini_llm::kernels {

bool can_use_decode_gemm(
    int M,
    int N,
    int K
);

void launch_decode_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    const float* bias,
    float* C,
    cudaStream_t stream = 0
);

} // namespace mini_llm::kernels