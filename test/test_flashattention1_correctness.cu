#include "kernel/flashattention1.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            std::exit(1);                                                   \
        }                                                                  \
    } while (0)

static void cpu_native_causal_attention(
    const std::vector<float>& qkv,
    std::vector<float>& out,
    int seq_len,
    int d_model,
    int d_head,
    int n_heads
) {
    const float scale = 1.0f / std::sqrt((float)d_head);

    std::fill(out.begin(), out.end(), 0.0f);

    for (int h = 0; h < n_heads; h++) {
        for (int qi = 0; qi < seq_len; qi++) {
            std::vector<float> scores(qi + 1);

            float max_score = -std::numeric_limits<float>::infinity();

            for (int kj = 0; kj <= qi; kj++) {
                float dot = 0.0f;

                const float* q_ptr =
                    qkv.data()
                    + (size_t)qi * 3 * d_model
                    + h * d_head;

                const float* k_ptr =
                    qkv.data()
                    + (size_t)kj * 3 * d_model
                    + d_model
                    + h * d_head;

                for (int d = 0; d < d_head; d++) {
                    dot += q_ptr[d] * k_ptr[d];
                }

                float score = dot * scale;
                scores[kj] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;

            for (int kj = 0; kj <= qi; kj++) {
                scores[kj] = std::exp(scores[kj] - max_score);
                denom += scores[kj];
            }

            for (int d = 0; d < d_head; d++) {
                float acc = 0.0f;

                for (int kj = 0; kj <= qi; kj++) {
                    const float* v_ptr =
                        qkv.data()
                        + (size_t)kj * 3 * d_model
                        + 2 * d_model
                        + h * d_head;

                    float prob = scores[kj] / denom;
                    acc += prob * v_ptr[d];
                }

                out[(size_t)qi * d_model + h * d_head + d] = acc;
            }
        }
    }
}

static bool run_one_test(
    int seq_len,
    int d_model,
    int d_head,
    int n_heads,
    unsigned int seed
) {
    size_t qkv_elems = (size_t)seq_len * 3 * d_model;
    size_t out_elems = (size_t)seq_len * d_model;

    std::vector<float> h_qkv(qkv_elems);
    std::vector<float> h_flash(out_elems);
    std::vector<float> h_ref(out_elems);

    /*
     * 너무 큰 random 값을 쓰면 exp overflow/오차가 커질 수 있으므로
     * 작은 값으로 생성한다.
     */
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (size_t i = 0; i < qkv_elems; i++) {
        h_qkv[i] = dist(rng);
    }

    float* d_qkv = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_qkv, qkv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_qkv,
        h_qkv.data(),
        qkv_elems * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(float)));

    const float scale = 1.0f / std::sqrt((float)d_head);

    flashattention1_prefill(
        d_qkv,
        d_out,
        seq_len,
        d_model,
        d_head,
        n_heads,
        scale
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_flash.data(),
        d_out,
        out_elems * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    cpu_native_causal_attention(
        h_qkv,
        h_ref,
        seq_len,
        d_model,
        d_head,
        n_heads
    );

    double max_abs = 0.0;
    double mean_abs = 0.0;
    double max_rel = 0.0;

    for (size_t i = 0; i < out_elems; i++) {
        double ref = (double)h_ref[i];
        double got = (double)h_flash[i];

        double abs_err = std::abs(ref - got);
        double rel_err = abs_err / (std::abs(ref) + 1e-8);

        max_abs = std::max(max_abs, abs_err);
        mean_abs += abs_err;
        max_rel = std::max(max_rel, rel_err);
    }

    mean_abs /= (double)out_elems;

    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_out));

    /*
     * FP32 online softmax와 CPU reference의 순서 차이를 고려한 tolerance.
     * random scale을 작게 줬으므로 보통 이보다 훨씬 작게 나온다.
     */
    const double max_abs_tol = 2e-4;
    const double mean_abs_tol = 2e-5;

    bool pass = (max_abs <= max_abs_tol) && (mean_abs <= mean_abs_tol);

    std::cout
        << "seq_len=" << seq_len
        << ", d_model=" << d_model
        << ", d_head=" << d_head
        << ", n_heads=" << n_heads
        << ", max_abs=" << max_abs
        << ", mean_abs=" << mean_abs
        << ", max_rel=" << max_rel
        << ", result=" << (pass ? "PASS" : "FAIL")
        << std::endl;

    return pass;
}

int main(int argc, char** argv) {
    /*
     * 기본값은 GPT-2 small 설정.
     *
     * 사용 예:
     *   ./test_flashattention1_correctness
     *   ./test_flashattention1_correctness 128
     */
    int d_model = 768;
    int d_head = 64;
    int n_heads = 12;

    std::vector<int> seq_lens = {
        1, 2, 4, 8, 16, 32, 64, 128
    };

    if (argc >= 2) {
        seq_lens.clear();
        seq_lens.push_back(std::atoi(argv[1]));
    }

    if (d_model != d_head * n_heads) {
        std::cerr << "Invalid config: d_model != d_head * n_heads" << std::endl;
        return 1;
    }

    bool all_pass = true;

    for (int seq_len : seq_lens) {
        bool pass = run_one_test(
            seq_len,
            d_model,
            d_head,
            n_heads,
            1234u + (unsigned int)seq_len
        );

        all_pass = all_pass && pass;
    }

    if (!all_pass) {
        std::cerr << "[FAIL] FlashAttention1 output differs from native attention." << std::endl;
        return 1;
    }

    std::cout << "[PASS] FlashAttention1 matches native causal attention." << std::endl;
    return 0;
}
