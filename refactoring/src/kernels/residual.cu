#include "kernels/residual.cuh"
#include <cuda_runtime.h>

namespace mini_llm::kernels {

__global__ void residual_add_kernel(
    float* __restrict__ x,
    const float* __restrict__ y,
    int nums
) {
    constexpr int VEC = 4;

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = vec_idx * VEC;

    if (base + 3 < nums) {
        float4 xv = reinterpret_cast<float4*>(x)[vec_idx];
        float4 yv = reinterpret_cast<const float4*>(y)[vec_idx];

        xv.x += yv.x;
        xv.y += yv.y;
        xv.z += yv.z;
        xv.w += yv.w;

        reinterpret_cast<float4*>(x)[vec_idx] = xv;
    } else {
        for (int i = base; i < nums; ++i) {
            x[i] += y[i];
        }
    }
}

void residual_add(
    float* x,
    const float* y,
    int nums
) {
    if (nums <= 0) {
        return;
    }

    constexpr int VEC = 4;
    int vec_nums = (nums + VEC - 1) / VEC;

    int threads = 256;
    int blocks = (vec_nums + threads - 1) / threads;

    residual_add_kernel<<<blocks, threads>>>(
        x,
        y,
        nums
    );
}

} // namespace mini_llm::kernels