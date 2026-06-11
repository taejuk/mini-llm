#include "test_common.cuh"

#include "constants.h"
#include "kernels/embedding.cuh"

int main() {
    namespace C = mini_llm::constants;

    constexpr int SEQ = 7;
    constexpr int VOCAB = 64;
    constexpr int MAX_POS = 32;
    constexpr int D = C::GPT2_D_MODEL;

    auto h_tokens = read_bin<int>(case_path("embedding", "tokens.bin"), SEQ);
    auto h_pos = read_bin<int>(case_path("embedding", "pos.bin"), SEQ);
    auto h_wte = read_bin<float>(case_path("embedding", "wte.bin"), VOCAB * D);
    auto h_wpe = read_bin<float>(case_path("embedding", "wpe.bin"), MAX_POS * D);
    auto expected = read_bin<float>(case_path("embedding", "expected.bin"), SEQ * D);

    int* d_tokens = nullptr;
    int* d_pos = nullptr;
    float* d_wte = nullptr;
    float* d_wpe = nullptr;
    float* d_x = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tokens, SEQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos, SEQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wte, VOCAB * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wpe, MAX_POS * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, SEQ * D * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), SEQ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), SEQ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wte, h_wte.data(), VOCAB * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wpe, h_wpe.data(), MAX_POS * D * sizeof(float), cudaMemcpyHostToDevice));

    mini_llm::kernels::embedding_lookup(
        d_tokens,
        d_pos,
        d_wte,
        d_wpe,
        d_x,
        SEQ
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(SEQ * D);
    CUDA_CHECK(cudaMemcpy(got.data(), d_x, SEQ * D * sizeof(float), cudaMemcpyDeviceToHost));

    check_close("embedding_lookup", got, expected, 1e-6f);

    cudaFree(d_tokens);
    cudaFree(d_pos);
    cudaFree(d_wte);
    cudaFree(d_wpe);
    cudaFree(d_x);

    return 0;
}