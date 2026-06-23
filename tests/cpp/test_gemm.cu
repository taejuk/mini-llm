#include "test_common.cuh"

#include "kernels/gemm.cuh"
#include "kernels/linear.cuh"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

float deterministic_value(size_t i, float scale = 0.03125f) {
    int v = static_cast<int>((i * 13 + 7) % 23) - 11;
    return static_cast<float>(v) * scale;
}

float gelu_ref(float v) {
    return 0.5f * v *
           (1.0f + std::tanh(
               0.7978845608f *
               (v + 0.044715f * v * v * v)
           ));
}

std::vector<float> make_vector(size_t n, float scale = 0.03125f) {
    std::vector<float> v(n);

    for (size_t i = 0; i < n; i++) {
        v[i] = deterministic_value(i, scale);
    }

    return v;
}

std::vector<float> reference_linear(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& bias,
    const std::vector<float>& C_init,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    bool apply_gelu
) {
    std::vector<float> out(static_cast<size_t>(M) * N);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;

            for (int k = 0; k < K; k++) {
                acc += A[static_cast<size_t>(m) * K + k] *
                       B[static_cast<size_t>(k) * N + n];
            }

            size_t idx = static_cast<size_t>(m) * N + n;
            float old = C_init.empty() ? 0.0f : C_init[idx];
            float b = bias.empty() ? 0.0f : bias[n];
            float value = alpha * acc + beta * old + b;

            out[idx] = apply_gelu ? gelu_ref(value) : value;
        }
    }

    return out;
}

void copy_to_device(float** dst, const std::vector<float>& src) {
    CUDA_CHECK(cudaMalloc(dst, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        *dst,
        src.data(),
        src.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

void check_gemm_bias_gelu(
    const std::string& name,
    int M,
    int N,
    int K,
    float tol
) {
    auto A = make_vector(static_cast<size_t>(M) * K, 0.01953125f);
    auto B = make_vector(static_cast<size_t>(K) * N, 0.015625f);
    auto bias = make_vector(N, 0.0078125f);
    std::vector<float> zero(static_cast<size_t>(M) * N, 0.0f);

    auto expected = reference_linear(
        A,
        B,
        bias,
        zero,
        M,
        N,
        K,
        1.0f,
        0.0f,
        true
    );

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_bias = nullptr;
    float* d_C = nullptr;

    copy_to_device(&d_A, A);
    copy_to_device(&d_B, B);
    copy_to_device(&d_bias, bias);
    CUDA_CHECK(cudaMalloc(&d_C, zero.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, zero.size() * sizeof(float)));

    mini_llm::kernels::launch_gemm_bias_gelu(
        M,
        N,
        K,
        d_A,
        d_B,
        d_bias,
        d_C
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(zero.size());
    CUDA_CHECK(cudaMemcpy(
        got.data(),
        d_C,
        got.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_close(name, got, expected, tol);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
}

void check_linear_gelu_decode_dispatch() {
    constexpr int M = 4;
    constexpr int K = 768;
    constexpr int N = 3072;

    auto A = make_vector(static_cast<size_t>(M) * K, 0.0078125f);
    auto B = make_vector(static_cast<size_t>(K) * N, 0.00390625f);
    auto bias = make_vector(N, 0.001953125f);
    std::vector<float> zero(static_cast<size_t>(M) * N, 0.0f);

    auto expected = reference_linear(
        A,
        B,
        bias,
        zero,
        M,
        N,
        K,
        1.0f,
        0.0f,
        true
    );

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_bias = nullptr;
    float* d_C = nullptr;

    copy_to_device(&d_A, A);
    copy_to_device(&d_B, B);
    copy_to_device(&d_bias, bias);
    CUDA_CHECK(cudaMalloc(&d_C, zero.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, zero.size() * sizeof(float)));

    mini_llm::kernels::linear_gelu(
        d_A,
        d_B,
        d_bias,
        d_C,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(zero.size());
    CUDA_CHECK(cudaMemcpy(
        got.data(),
        d_C,
        got.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_close("linear_gelu_decode_dispatch", got, expected, 2e-2f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
}

void check_linear_residual_add() {
    constexpr int M = 5;
    constexpr int K = 7;
    constexpr int N = 11;

    auto A = make_vector(static_cast<size_t>(M) * K, 0.03125f);
    auto B = make_vector(static_cast<size_t>(K) * N, 0.025f);
    auto bias = make_vector(N, 0.0125f);
    auto C_init = make_vector(static_cast<size_t>(M) * N, 0.02f);

    auto expected = reference_linear(
        A,
        B,
        bias,
        C_init,
        M,
        N,
        K,
        1.0f,
        1.0f,
        false
    );

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_bias = nullptr;
    float* d_C = nullptr;

    copy_to_device(&d_A, A);
    copy_to_device(&d_B, B);
    copy_to_device(&d_bias, bias);
    copy_to_device(&d_C, C_init);

    mini_llm::kernels::linear_residual_add(
        d_A,
        d_B,
        d_bias,
        d_C,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(C_init.size());
    CUDA_CHECK(cudaMemcpy(
        got.data(),
        d_C,
        got.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_close("linear_residual_add", got, expected, 1e-3f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
}

void check_existing_gemm_case() {
    constexpr int ROWS = 67;
    constexpr int COLS = 79;
    constexpr int K = 37;
    constexpr float ALPHA = 0.5f;
    constexpr float BETA = 0.25f;

    auto h_A = read_bin<float>(case_path("gemm", "input_x.bin"), ROWS * K);
    auto h_B = read_bin<float>(case_path("gemm", "input_y.bin"), K * COLS);
    auto h_C = read_bin<float>(case_path("gemm", "input_z.bin"), ROWS * COLS);
    auto expected = read_bin<float>(case_path("gemm", "expected.bin"), ROWS * COLS);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, ROWS * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, ROWS * COLS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), ROWS * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));

    mini_llm::kernels::launch_gemm(
        ROWS,
        COLS,
        K,
        ALPHA,
        d_A,
        d_B,
        BETA,
        d_C
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(ROWS * COLS);
    CUDA_CHECK(cudaMemcpy(got.data(), d_C, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

    check_close("gemm", got, expected, 1e-3f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // namespace

int main() {
    // Force custom kernels so the fusion tests exercise GEMM epilogue and
    // decode_gemm_gelu dispatch rather than the cuBLAS fallback path.
    unsetenv("MINI_LLM_USE_CUBLAS");

    check_existing_gemm_case();

    // Non-tile-aligned shape: exercises gemm_general_kernel with GELU epilogue.
    check_gemm_bias_gelu(
        "gemm_bias_gelu_general",
        9,
        73,
        37,
        1e-3f
    );

    // Tile-aligned shape: exercises gemm_vectorized_bank_kernel with GELU epilogue.
    check_gemm_bias_gelu(
        "gemm_bias_gelu_vectorized",
        64,
        64,
        8,
        1e-3f
    );

    // GPT-2 fc1 decode shape: exercises linear_gelu -> decode_gemm_gelu.
    check_linear_gelu_decode_dispatch();

    // Tests linear + residual-add fusion path.
    check_linear_residual_add();

    return 0;
}
