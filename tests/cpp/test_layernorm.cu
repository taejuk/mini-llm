#include "test_common.cuh"

#include "kernels/layernorm.cuh"

int main() {
    constexpr int ROWS = 17;
    constexpr int COLS = 768;

    auto h_x = read_bin<float>(case_path("layernorm", "input_x.bin"), ROWS * COLS);
    auto h_gamma = read_bin<float>(case_path("layernorm", "input_gamma.bin"), COLS);
    auto h_beta = read_bin<float>(case_path("layernorm", "input_beta.bin"), COLS);
    auto expected = read_bin<float>(case_path("layernorm", "expected.bin"), ROWS * COLS);

    float* d_x = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, ROWS * COLS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));

    mini_llm::kernels::layernorm(
        d_x,
        d_gamma,
        d_beta,
        d_y,
        ROWS,
        COLS
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(ROWS * COLS);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

    check_close("layernorm", got, expected, 1e-4f);

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_y);

    return 0;
}