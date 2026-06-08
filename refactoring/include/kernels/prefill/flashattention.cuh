#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

#include "constants.h"
#include "kernels/warp_reduction.cuh"
#define Br 4
#define Bc 32

namespace mini_llm::kernels {
void flashattention_prefill(
    const float* buf_qkv,
    float* buf_O,
    int seq_len,
    float scale
);
}