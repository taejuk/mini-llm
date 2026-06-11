#include "kernels/embedding.cuh"

namespace mini_llm::kernels {
__global__ void embedding_kernel(
    const int* __restrict__ ids,
    const int* __restrict__ pos,
    const float* __restrict__ wte,
    const float* __restrict__ wpe,
    float* __restrict__ x,
    int seq_len
) {
    constexpr int D_MODEL = mini_llm::constants::GPT2_D_MODEL;
    constexpr int VEC = 4;
    constexpr int VEC_PER_TOKEN = D_MODEL / VEC;

    int r = blockIdx.x;
    int v = threadIdx.x;

    if (r >= seq_len || v >= VEC_PER_TOKEN) {
        return;
    }

    int token_id = ids[r];
    int position = pos[r];

    const float* wte_row = wte + token_id * D_MODEL;
    const float* wpe_row = wpe + position * D_MODEL;
    float* x_row = x + r * D_MODEL;

    const float4* wte4 = reinterpret_cast<const float4*>(wte_row);
    const float4* wpe4 = reinterpret_cast<const float4*>(wpe_row);
    float4* x4 = reinterpret_cast<float4*>(x_row);

    float4 a = wte4[v];
    float4 b = wpe4[v];

    float4 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    out.w = a.w + b.w;

    x4[v] = out;
}

void embedding_lookup(
    const int* d_token_ids,
    const int* d_pos,
    const float* wte,
    const float* wpe,
    float* x,
    int seq_len
) {
    constexpr int D_MODEL = mini_llm::constants::GPT2_D_MODEL;
    static_assert(D_MODEL % 4 == 0, "D_MODEL must be divisible by 4");

    dim3 grid(seq_len);
    dim3 block(D_MODEL / 4); 

    embedding_kernel<<<grid, block>>>(
        d_token_ids,
        d_pos,
        wte,
        wpe,
        x,
        seq_len
    );
}
}