#pragma once
#include <cuda_runtime.h>

namespace mini_llm::kernels {
void append_prefill_kv(
    const float* buf_qkv,
    float* pool,
    const int* d_token_to_block,
    const int* d_token_to_offset,
    int seq_len
);
}