#include "kernels/linear.cuh"

#include "kernels/gemm.cuh"
#include "constants.h"

namespace mini_llm::kernels {

void linear(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int M,
    int K,
    int N
) {
    launch_gemm_bias(
        M,
        N,
        K,
        x,
        weight,
        bias,
        y
    );
}

void linear_residual_add(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int M,
    int K,
    int N
) {
    launch_gemm_bias_residual(
        M,
        N,
        K,
        x,
        weight,
        bias,
        y
    );
}

void linear_gelu(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int M,
    int K,
    int N
) {
    launch_gemm_bias_gelu(
        M,
        N,
        K,
        x,
        weight,
        bias,
        y
    );
}


void qkv_projection(
    const float* buf_ln,
    const float* qkv_w,
    const float* qkv_b,
    float* buf_qkv,
    int num_rows
) {
    constexpr int D = mini_llm::constants::GPT2_D_MODEL;

    linear(
        buf_ln,
        qkv_w,
        qkv_b,
        buf_qkv,
        num_rows,
        D,
        3 * D
    );
}

} // namespace mini_llm::kernels