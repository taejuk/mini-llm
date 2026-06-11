#pragma once

#include <cuda_runtime.h>

namespace mini_llm::kernels {

void paged_decode_attention(
    const float* buf_qkv,          // [B, 3D]
    const int* d_block_table,      // flattened block tables
    const int* d_block_offsets,    // [B]
    const int* d_num_tokens,       // [B]
    const float* pool,
    float* buf_O,                 // [B, D]
    int batch_size,
    int max_seq
);

} // namespace mini_llm::kernels