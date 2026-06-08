#pragma once
#include <cuda_runtime.h>

namespace mini_llm::kernels {
__inline__ __device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;

    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if(lane == 0) shared[warp_id] = val;
    __syncthreads();

    if(threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + 31) >> 5;
        for(int i = 0; i < num_warps; i++) total += shared[i];
        shared[0] = total;
    }

    __syncthreads();
    return shared[0];
}
}