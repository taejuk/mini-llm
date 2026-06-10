#include "test_common.cuh"

#include "kernels/gelu.cuh"

int main() {
    constexpr int N = 1027;

    auto h_x = read_bin<float>(case_path("gelu", "input_x.bin"), N);
    auto expected = read_bin<float>(case_path("gelu", "expected.bin"), N);

    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    mini_llm::kernels::gelu(d_x, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

    check_close("gelu", got, expected, 1e-5f);

    cudaFree(d_x);
    return 0;
}