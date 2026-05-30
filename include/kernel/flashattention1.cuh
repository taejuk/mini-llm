#pragma once

#include <cuda_runtime.h>

void flashattention1_prefill(
    const float* buf_qkv,
    float* buf_O,
    int seq_len,
    int d_model,
    int d_head,
    int n_heads,
    float scale
);
