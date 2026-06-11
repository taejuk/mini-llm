#pragma once
#include <cuda_runtime.h>

namespace mini_llm::kernels {

__inline__ __device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + 31) >> 5;

        for (int i = 0; i < num_warps; i++) {
            total += shared[i];
        }

        shared[0] = total;
    }

    __syncthreads();

    return shared[0];
}

__inline__ __device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        val = fmaxf(val, other);
    }

    return val;
}

__inline__ __device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float max_val = shared[0];
        int num_warps = (blockDim.x + 31) >> 5;

        for (int i = 1; i < num_warps; i++) {
            max_val = fmaxf(max_val, shared[i]);
        }

        shared[0] = max_val;
    }

    __syncthreads();

    return shared[0];
}

} // namespace mini_llm::kernels