#include "test_common.cuh"

#include "kernels/residual.cuh"

int main() {
    constexpr int N = 1027;

    auto h_x = read_bin<float>(case_path("residual", "input_x.bin"), N);
    auto h_y = read_bin<float>(case_path("residual", "input_y.bin"), N);
    auto expected = read_bin<float>(case_path("residual", "expected.bin"), N);

    float* d_x = nullptr;
    float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    mini_llm::kernels::residual_add(d_x, d_y, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

    check_close("residual_add", got, expected, 1e-6f);

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}