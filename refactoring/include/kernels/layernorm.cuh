#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "kernels/warp_reduction.cuh"

namespace mini_llm::kernels {
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int num_rows,
    float eps
);

void layernorm(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int num_rows,
    float eps = 1e-5f
);

}