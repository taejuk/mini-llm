#include "kernels/layernorm.cuh"

namespace mini_llm::kernels {
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int num_rows,
    float eps
) {
    constexpr int D = mini_llm::constants::GPT2_D_MODEL;
    constexpr int VEC = 4;
    //constexpr int VEC_PER_ROW = D / VEC;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    const float4* x4 = reinterpret_cast<const float4*>(x_row);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    float4* y4 = reinterpret_cast<float4*>(y_row);

    float4 v = x4[tid];
    float4 g = gamma4[tid];
    float4 b = beta4[tid];
    float4 out;

    float local_sum = v.x + v.y + v.z + v.w;
    float sum = block_reduce_sum(local_sum);
    float mean = sum / static_cast<float>(D);

    float dx0 = v.x - mean;
    float dx1 = v.y - mean;
    float dx2 = v.z - mean;
    float dx3 = v.w - mean;
    float local_var_sum = dx0*dx0 + dx1*dx1 + dx2*dx2 + dx3*dx3;

    float var_sum = block_reduce_sum(local_var_sum);
    float variance = var_sum / static_cast<float>(D);
    float inv_std = rsqrtf(variance + eps);

    out.x = (v.x - mean) * inv_std * g.x + b.x;
    out.y = (v.y - mean) * inv_std * g.y + b.y;
    out.z = (v.z - mean) * inv_std * g.z + b.z;
    out.w = (v.w - mean) * inv_std * g.w + b.w;

    y4[tid] = out;
}

void layernorm(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int num_rows,
    float eps = 1e-5f
) {
    constexpr int D = mini_llm::constants::GPT2_D_MODEL;
    constexpr int VEC = 4;
    constexpr int THREADS = D / VEC;

    static_assert(D % VEC == 0, "D_MODEL must be divisible by 4");

    if (num_rows <= 0) {
        return;
    }

    dim3 grid(num_rows);
    dim3 block(THREADS);

    layernorm_kernel<<<grid, block, 0, stream>>>(
        x,
        gamma,
        beta,
        y,
        num_rows,
        eps
    );
}

}