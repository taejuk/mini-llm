#include "kernels/gelu.cuh"
#include <cuda_runtime.h>
#include <math.h>

namespace mini_llm::kernels {

__device__ __forceinline__ float gelu_one(float v) {
    return 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f*v*v*v)));
}


__global__ void gelu_kernel(float* x, int n) {
    constexpr int VEC = 4;
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = vec_idx * VEC;

    if(base + 3 < n) {
        float4 xv = reinterpret_cast<float4*>(x)[vec_idx];
        xv.x = gelu_one(xv.x);
        xv.y = gelu_one(xv.y);
        xv.z = gelu_one(xv.z);
        xv.w = gelu_one(xv.w);
    } else {
        for(int i = base; i < n; i++) {
            x[i] = gelu_one(x[i]);
        }
    }
}

void gelu(float* x, int nums) {
    if (nums <= 0) {
        return;
    }

    constexpr int VEC = 4;
    int vec_nums = (nums + VEC - 1) / VEC;

    int threads = 256;
    int blocks = (vec_nums + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(
        x,
        nums
    );
}
}