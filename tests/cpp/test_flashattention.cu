#include "test_common.cuh"

#include "constants.h"
#include "kernels/prefill/flashattention.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace C = mini_llm::constants;

namespace {

void fill_qkv(std::vector<float>& qkv, int seq_len) {
    constexpr int D = C::GPT2_D_MODEL;

    for (int t = 0; t < seq_len; ++t) {
        size_t base = static_cast<size_t>(t) * 3 * D;

        for (int i = 0; i < D; ++i) {
            float tf = static_cast<float>(t + 1);
            float df = static_cast<float>(i + 1);

            // Small deterministic values keep softmax numerically stable while
            // still producing non-trivial head/token-dependent outputs.
            qkv[base + i] =
                0.20f * std::sin(0.013f * tf + 0.007f * df);

            qkv[base + D + i] =
                0.20f * std::cos(0.011f * (tf + 1.0f) - 0.005f * (df + 2.0f));

            qkv[base + 2 * D + i] =
                0.10f * std::sin(0.017f * (tf + 2.0f) + 0.003f * (df + 4.0f));
        }
    }
}

std::vector<float> reference_causal_attention(
    const std::vector<float>& qkv,
    int seq_len,
    float scale
) {
    constexpr int H = C::GPT2_N_HEADS;
    constexpr int D = C::GPT2_D_MODEL;
    constexpr int Dh = C::GPT2_D_HEAD;

    std::vector<float> out(static_cast<size_t>(seq_len) * D, 0.0f);
    std::vector<double> scores(seq_len, 0.0);

    for (int h = 0; h < H; ++h) {
        for (int q = 0; q < seq_len; ++q) {
            double max_score = -std::numeric_limits<double>::infinity();

            for (int k = 0; k <= q; ++k) {
                double dot = 0.0;

                for (int d = 0; d < Dh; ++d) {
                    size_t q_idx =
                        static_cast<size_t>(q) * 3 * D +
                        h * Dh +
                        d;

                    size_t k_idx =
                        static_cast<size_t>(k) * 3 * D +
                        D +
                        h * Dh +
                        d;

                    dot += static_cast<double>(qkv[q_idx]) *
                           static_cast<double>(qkv[k_idx]);
                }

                scores[k] = dot * static_cast<double>(scale);
                max_score = std::max(max_score, scores[k]);
            }

            double denom = 0.0;
            for (int k = 0; k <= q; ++k) {
                denom += std::exp(scores[k] - max_score);
            }

            for (int d = 0; d < Dh; ++d) {
                double acc = 0.0;

                for (int k = 0; k <= q; ++k) {
                    double p = std::exp(scores[k] - max_score) / denom;

                    size_t v_idx =
                        static_cast<size_t>(k) * 3 * D +
                        2 * D +
                        h * Dh +
                        d;

                    acc += p * static_cast<double>(qkv[v_idx]);
                }

                out[static_cast<size_t>(q) * D + h * Dh + d] =
                    static_cast<float>(acc);
            }
        }
    }

    return out;
}

} // namespace

int main() {
    constexpr int SEQ = 37; // Covers partial Br tile and multiple Bc tiles.
    constexpr int D = C::GPT2_D_MODEL;
    constexpr float SCALE = 1.0f / 8.0f; // 1 / sqrt(GPT2_D_HEAD=64)

    std::vector<float> h_qkv(static_cast<size_t>(SEQ) * 3 * D);
    fill_qkv(h_qkv, SEQ);

    auto expected = reference_causal_attention(h_qkv, SEQ, SCALE);

    float* d_qkv = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_qkv, h_qkv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, expected.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_qkv,
        h_qkv.data(),
        h_qkv.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemset(d_out, 0, expected.size() * sizeof(float)));

    mini_llm::kernels::flashattention_prefill(
        d_qkv,
        d_out,
        SEQ,
        SCALE
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(expected.size());
    CUDA_CHECK(cudaMemcpy(
        got.data(),
        d_out,
        got.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    check_close("flashattention_prefill", got, expected, 1e-4f);

    cudaFree(d_qkv);
    cudaFree(d_out);

    return 0;
}
