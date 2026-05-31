#include "kernel/flashattention1.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifndef FA1_BLOCK_M
#define FA1_BLOCK_M 0
#endif

#ifndef FA1_BLOCK_N
#define FA1_BLOCK_N 0
#endif

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            std::exit(1);                                                   \
        }                                                                  \
    } while (0)

static double mean(std::vector<double> v) {
    if (v.empty()) return 0.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / (double)v.size();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;

    std::sort(v.begin(), v.end());

    double idx = p * (double)(v.size() - 1);
    int lo = (int)std::floor(idx);
    int hi = (int)std::ceil(idx);

    if (lo == hi) return v[lo];

    double t = idx - (double)lo;
    return v[lo] * (1.0 - t) + v[hi] * t;
}

/*
 * Deterministic pseudo-random initialization.
 *
 * 목적:
 *   host에서 큰 qkv 배열을 만들고 복사하지 않고,
 *   GPU에서 바로 buf_qkv를 채운다.
 *
 * qkv 값은 작은 범위로 둔다.
 * attention score가 너무 커지면 exp overflow나 softmax saturation이 생길 수 있다.
 */
__global__ void init_qkv_kernel(float* qkv, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int x = (unsigned int)(idx * 1664525u + 1013904223u);
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;

    float u = (float)(x & 0xffffu) / 65535.0f;
    qkv[idx] = (u - 0.5f) * 0.04f;
}

static void init_qkv(float* d_qkv, size_t n) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);

    init_qkv_kernel<<<blocks, threads>>>(d_qkv, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " warmup_runs measure_runs prompt_len...\n\n"
        << "Example:\n"
        << "  " << prog << " 5 30 32 64 128 256 512 1024\n\n"
        << "Build-time options:\n"
        << "  -DFA1_BLOCK_M=4\n"
        << "  -DFA1_BLOCK_N=32\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    int warmup_runs = std::atoi(argv[1]);
    int measure_runs = std::atoi(argv[2]);

    if (warmup_runs < 0 || measure_runs <= 0) {
        std::cerr << "Invalid warmup_runs or measure_runs\n";
        return 1;
    }

    std::vector<int> prompt_lens;
    for (int i = 3; i < argc; i++) {
        int v = std::atoi(argv[i]);
        if (v <= 0) {
            std::cerr << "Invalid prompt_len: " << argv[i] << "\n";
            return 1;
        }
        prompt_lens.push_back(v);
    }

    int max_prompt_len = *std::max_element(prompt_lens.begin(), prompt_lens.end());

    /*
     * GPT-2 small config.
     */
    constexpr int D_MODEL = 768;
    constexpr int N_HEADS = 12;
    constexpr int D_HEAD = 64;

    static_assert(D_MODEL == N_HEADS * D_HEAD, "Invalid GPT-2 config");

    const float scale = 1.0f / std::sqrt((float)D_HEAD);

    size_t max_qkv_elems = (size_t)max_prompt_len * 3 * D_MODEL;
    size_t max_o_elems = (size_t)max_prompt_len * D_MODEL;

    float* d_qkv = nullptr;
    float* d_O = nullptr;

    CUDA_CHECK(cudaMalloc(&d_qkv, max_qkv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, max_o_elems * sizeof(float)));

    init_qkv(d_qkv, max_qkv_elems);
    CUDA_CHECK(cudaMemset(d_O, 0, max_o_elems * sizeof(float)));

    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::cout
        << "[config] benchmark=flashattention1_only"
        << " FA1_BLOCK_M=" << FA1_BLOCK_M
        << " FA1_BLOCK_N=" << FA1_BLOCK_N
        << " warmup_runs=" << warmup_runs
        << " measure_runs=" << measure_runs
        << " max_prompt_len=" << max_prompt_len
        << " d_model=" << D_MODEL
        << " n_heads=" << N_HEADS
        << " d_head=" << D_HEAD
        << "\n";

    /*
     * CSV header.
     *
     * qk_tflops는 QK^T dot product 기준의 rough estimate다.
     * FlashAttention은 softmax와 O update도 수행하므로 전체 FLOPs는 이것보다 크다.
     */
    std::cout
        << "attention_impl,"
        << "fa1_block_m,"
        << "fa1_block_n,"
        << "seq_len,"
        << "gpu_ms_mean,"
        << "gpu_ms_p50,"
        << "gpu_ms_p95,"
        << "prefill_tok_per_s,"
        << "output_elem_per_s,"
        << "qk_flops_est,"
        << "qk_tflops_est,"
        << "sample_out"
        << "\n";

    for (int seq_len : prompt_lens) {
        size_t qkv_elems = (size_t)seq_len * 3 * D_MODEL;
        size_t o_elems = (size_t)seq_len * D_MODEL;

        CUDA_CHECK(cudaMemset(d_O, 0, o_elems * sizeof(float)));

        /*
         * Warmup.
         */
        for (int i = 0; i < warmup_runs; i++) {
            flashattention1_prefill(
                d_qkv,
                d_O,
                seq_len,
                D_MODEL,
                D_HEAD,
                N_HEADS,
                scale
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> gpu_times;
        gpu_times.reserve(measure_runs);

        /*
         * Measurement.
         */
        for (int i = 0; i < measure_runs; i++) {
            CUDA_CHECK(cudaEventRecord(ev_start));

            flashattention1_prefill(
                d_qkv,
                d_O,
                seq_len,
                D_MODEL,
                D_HEAD,
                N_HEADS,
                scale
            );

            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

            gpu_times.push_back((double)ms);
        }

        /*
         * 결과가 실제로 쓰였는지 확인하기 위한 작은 sample read.
         * timing에는 포함하지 않는다.
         */
        float sample_out = 0.0f;
        CUDA_CHECK(cudaMemcpy(
            &sample_out,
            d_O + (o_elems > 0 ? o_elems - 1 : 0),
            sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        double gpu_ms_mean = mean(gpu_times);
        double gpu_ms_p50 = percentile(gpu_times, 0.50);
        double gpu_ms_p95 = percentile(gpu_times, 0.95);

        double prefill_tok_per_s =
            gpu_ms_mean > 0.0
            ? (double)seq_len * 1000.0 / gpu_ms_mean
            : 0.0;

        double output_elem_per_s =
            gpu_ms_mean > 0.0
            ? (double)o_elems * 1000.0 / gpu_ms_mean
            : 0.0;

        /*
         * Causal QK dot product FLOPs estimate.
         *
         * pair_count = seq_len * (seq_len + 1) / 2
         * each pair per head:
         *   dot product over D_HEAD = roughly 2 * D_HEAD FLOPs
         *
         * total:
         *   N_HEADS * pair_count * 2 * D_HEAD
         */
        double pair_count = (double)seq_len * (double)(seq_len + 1) / 2.0;
        double qk_flops_est = (double)N_HEADS * pair_count * 2.0 * (double)D_HEAD;

        double qk_tflops_est =
            gpu_ms_mean > 0.0
            ? qk_flops_est / (gpu_ms_mean / 1000.0) / 1.0e12
            : 0.0;

        std::cout
            << "flashattention1,"
            << FA1_BLOCK_M << ","
            << FA1_BLOCK_N << ","
            << seq_len << ","
            << gpu_ms_mean << ","
            << gpu_ms_p50 << ","
            << gpu_ms_p95 << ","
            << prefill_tok_per_s << ","
            << output_elem_per_s << ","
            << qk_flops_est << ","
            << qk_tflops_est << ","
            << sample_out
            << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    CUDA_CHECK(cudaFree(d_qkv));
    CUDA_CHECK(cudaFree(d_O));

    return 0;
}
