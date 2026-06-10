#include "test_common.cuh"

#include "kernels/gemm.cuh"

int main() {
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

    return 0;
}