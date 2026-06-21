#include "kernels/argmax.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace mini_llm::kernels {

namespace {

constexpr int ARGMAX_THREADS = 256;

__device__ __forceinline__ bool better_pair(
    float new_val,
    int new_idx,
    float old_val,
    int old_idx
) {
    return (new_val > old_val) ||
           (new_val == old_val && new_idx < old_idx);
}

__global__ void argmax_rows_kernel(
    const float* __restrict__ logits,  // [rows, cols]
    int* __restrict__ output_tokens,   // [rows]
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float best_val = -CUDART_INF_F;
    int best_idx = 0;

    const float* row_ptr =
        logits + static_cast<size_t>(row) * static_cast<size_t>(cols);

    for (int col = tid; col < cols; col += blockDim.x) {
        float v = row_ptr[col];

        if (better_pair(v, col, best_val, best_idx)) {
            best_val = v;
            best_idx = col;
        }
    }

    __shared__ float s_val[ARGMAX_THREADS];
    __shared__ int s_idx[ARGMAX_THREADS];

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;

    __syncthreads();

    for (int stride = ARGMAX_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_val[tid + stride];
            int other_idx = s_idx[tid + stride];

            if (better_pair(other_val, other_idx, s_val[tid], s_idx[tid])) {
                s_val[tid] = other_val;
                s_idx[tid] = other_idx;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        output_tokens[row] = s_idx[0];
    }
}

} // anonymous namespace

void argmax_gpu(
    const float* logits,
    int* output_tokens,
    int rows,
    int cols,
    cudaStream_t stream
) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    dim3 grid(rows);
    dim3 block(ARGMAX_THREADS);

    argmax_rows_kernel<<<grid, block, 0, stream>>>(
        logits,
        output_tokens,
        rows,
        cols
    );
}

} // namespace mini_llm::kernels