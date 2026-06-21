#pragma once

#include <cuda_runtime.h>

namespace mini_llm::kernels {

void argmax_gpu(
    const float* logits,
    int* output_tokens,
    int rows,
    int cols,
    cudaStream_t stream = 0
);

} // namespace mini_llm::kernels