#pragma once
#include <cuda_runtime.h>
#include <math.h>

#include "constants.h"

namespace mini_llm::kernels {
__global__ void embedding_kernel(
    const int* __restrict__ ids,
    const int* __restrict__ pos,
    const float* __restrict__ wte,
    const float* __restrict__ wpe,
    float* __restrict__ x,
    int seq_len
);

void embedding_lookup(
    const int* d_token_ids,
    const int* d_pos,
    const float* wte,
    const float* wpe,
    float* x,
    int seq_len
);
}