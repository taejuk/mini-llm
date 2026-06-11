#pragma once

#include <cuda_runtime.h>

namespace mini_llm::kernels {

void linear(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int M,
    int K,
    int N
);

void qkv_projection(
    const float* buf_ln,
    const float* qkv_w,
    const float* qkv_b,
    float* buf_qkv,
    int num_rows
);

}
