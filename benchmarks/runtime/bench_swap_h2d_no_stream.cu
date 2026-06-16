#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(err) << "\n";        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace {

constexpr int KV_BLOCK_SIZE = 16;
constexpr int GPT2_D_MODEL = 768;
constexpr int KV_PAIR = 2;  // K + V

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = idx - static_cast<double>(lo);

    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

} // namespace

int main(int argc, char** argv) {
    int max_blocks = 1024;
    int iterations = 1000;
    int warmup = 100;

    if (argc >= 2) {
        max_blocks = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        iterations = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        warmup = std::atoi(argv[3]);
    }

    const size_t block_elems =
        static_cast<size_t>(KV_BLOCK_SIZE) *
        static_cast<size_t>(GPT2_D_MODEL) *
        static_cast<size_t>(KV_PAIR);

    const size_t block_bytes = block_elems * sizeof(float);
    const size_t total_bytes = static_cast<size_t>(max_blocks) * block_bytes;
    const size_t total_elems = total_bytes / sizeof(float);

    std::cout << "KV_BLOCK_SIZE=" << KV_BLOCK_SIZE << "\n";
    std::cout << "GPT2_D_MODEL=" << GPT2_D_MODEL << "\n";
    std::cout << "KV_PAIR=" << KV_PAIR << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "max_blocks=" << max_blocks << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n\n";

    float* h_src = nullptr;
    float* d_dst = nullptr;

    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_src),
        total_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst),
        total_bytes
    ));

    for (size_t i = 0; i < total_elems; ++i) {
        h_src[i] = static_cast<float>(i % 1024);
    }

    std::cout
        << "direction,mode,blocks,copy_calls,bytes,total_avg_ms,total_p50_ms,"
        << "total_p99_ms,per_block_avg_us,bandwidth_GBps\n";

    for (int blocks = 1; blocks <= max_blocks; blocks *= 2) {
        const size_t bytes = static_cast<size_t>(blocks) * block_bytes;

        for (int i = 0; i < warmup; ++i) {
            for (int b = 0; b < blocks; ++b) {
                float* src = h_src + static_cast<size_t>(b) * block_elems;
                float* dst = d_dst + static_cast<size_t>(b) * block_elems;

                CUDA_CHECK(cudaMemcpy(
                    dst,
                    src,
                    block_bytes,
                    cudaMemcpyHostToDevice
                ));
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times_ms;
        times_ms.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < blocks; ++b) {
                float* src = h_src + static_cast<size_t>(b) * block_elems;
                float* dst = d_dst + static_cast<size_t>(b) * block_elems;

                CUDA_CHECK(cudaMemcpy(
                    dst,
                    src,
                    block_bytes,
                    cudaMemcpyHostToDevice
                ));
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            auto end = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(
                end - start
            ).count();

            times_ms.push_back(ms);
        }

        double sum = std::accumulate(
            times_ms.begin(),
            times_ms.end(),
            0.0
        );

        double avg_ms = sum / static_cast<double>(times_ms.size());
        double p50_ms = percentile(times_ms, 50.0);
        double p99_ms = percentile(times_ms, 99.0);

        double per_block_avg_us =
            (avg_ms * 1000.0) / static_cast<double>(blocks);

        double gb = static_cast<double>(bytes) / 1e9;
        double sec = avg_ms / 1000.0;
        double bandwidth_gbps = gb / sec;

        std::cout
            << "H2D,"
            << "sequential_cudaMemcpy,"
            << blocks << ","
            << blocks << ","
            << bytes << ","
            << avg_ms << ","
            << p50_ms << ","
            << p99_ms << ","
            << per_block_avg_us << ","
            << bandwidth_gbps
            << "\n";
    }

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_src));

    return 0;
}