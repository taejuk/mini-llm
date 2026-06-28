#include "constants.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace C = mini_llm::constants;

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

constexpr int BLOCK_COUNTS[] = {
    16, 64, 256, 1024
};

struct Result {
    double avg_ms = 0.0;
    double p50_ms = 0.0;
    double p99_ms = 0.0;
};

double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

double avg(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

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

Result summarize(const std::vector<double>& samples) {
    return Result{
        avg(samples),
        percentile(samples, 50.0),
        percentile(samples, 99.0)
    };
}

std::vector<int> make_destination_blocks(int blocks, std::mt19937& rng) {
    std::vector<int> dst(blocks);
    std::iota(dst.begin(), dst.end(), 0);
    std::shuffle(dst.begin(), dst.end(), rng);
    return dst;
}

void fill_host_blocks(float* h_pool, int blocks, size_t block_elems) {
    size_t total_elems = static_cast<size_t>(blocks) * block_elems;

    for (size_t i = 0; i < total_elems; ++i) {
        h_pool[i] = static_cast<float>(i % 1024);
    }
}

void copy_blocks_direct_h2d(
    float* d_pool,
    const float* h_pool,
    const std::vector<int>& dst_blocks,
    int blocks,
    size_t block_elems,
    size_t block_bytes
) {
    for (int i = 0; i < blocks; ++i) {
        const float* src =
            h_pool + static_cast<size_t>(i) * block_elems;

        float* dst =
            d_pool + static_cast<size_t>(dst_blocks[i]) * block_elems;

        CUDA_CHECK(cudaMemcpy(
            dst,
            src,
            block_bytes,
            cudaMemcpyHostToDevice
        ));
    }
}

double run_direct_swap_in_once(
    float* d_pool,
    const float* h_pool,
    const std::vector<int>& dst_blocks,
    int blocks,
    size_t block_elems,
    size_t block_bytes
) {
    CUDA_CHECK(cudaDeviceSynchronize());

    double t0 = now_ms();

    copy_blocks_direct_h2d(
        d_pool,
        h_pool,
        dst_blocks,
        blocks,
        block_elems,
        block_bytes
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    double t1 = now_ms();

    return t1 - t0;
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 100;
    int warmup = 10;
    unsigned int seed = 1234;

    if (argc >= 2) {
        iterations = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        warmup = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        seed = static_cast<unsigned int>(std::atoi(argv[3]));
    }

    if (iterations <= 0 || warmup < 0) {
        std::cerr
            << "usage: ./bench_direct_swap_in_by_blocks "
            << "[iterations] [warmup] [seed]\n";
        return 1;
    }

    const size_t block_elems =
        static_cast<size_t>(C::DEFAULT_KV_BLOCK_SIZE) *
        static_cast<size_t>(C::GPT2_D_MODEL) *
        2;

    const size_t block_bytes = block_elems * sizeof(float);

    int max_blocks = 0;
    for (int b : BLOCK_COUNTS) {
        max_blocks = std::max(max_blocks, b);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "block_elems=" << block_elems << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "NOTE: direct swap_in uses one cudaMemcpy H2D call per KV block.\n";
    std::cout << "NOTE: no BlockManager, no staging buffer, no scatter kernel.\n\n";

    float* h_pool = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_pool),
        static_cast<size_t>(max_blocks) * block_bytes
    ));

    fill_host_blocks(h_pool, max_blocks, block_elems);

    float* d_pool = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_pool),
        static_cast<size_t>(max_blocks) * block_bytes
    ));

    std::mt19937 rng(seed);

    std::cout
        << "blocks,bytes,"
        << "direct_swap_in_avg_ms,direct_swap_in_p50_ms,direct_swap_in_p99_ms,"
        << "avg_us_per_block,p50_us_per_block,p99_us_per_block,"
        << "effective_gbps\n";

    for (int blocks : BLOCK_COUNTS) {
        size_t bytes = static_cast<size_t>(blocks) * block_bytes;
        std::vector<int> dst_blocks = make_destination_blocks(blocks, rng);

        for (int i = 0; i < warmup; ++i) {
            (void)run_direct_swap_in_once(
                d_pool,
                h_pool,
                dst_blocks,
                blocks,
                block_elems,
                block_bytes
            );
        }

        std::vector<double> samples;
        samples.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            samples.push_back(run_direct_swap_in_once(
                d_pool,
                h_pool,
                dst_blocks,
                blocks,
                block_elems,
                block_bytes
            ));
        }

        Result res = summarize(samples);

        double avg_us_per_block =
            (res.avg_ms * 1000.0) / static_cast<double>(blocks);

        double p50_us_per_block =
            (res.p50_ms * 1000.0) / static_cast<double>(blocks);

        double p99_us_per_block =
            (res.p99_ms * 1000.0) / static_cast<double>(blocks);

        double effective_gbps =
            res.avg_ms > 0.0
                ? (static_cast<double>(bytes) / 1.0e9) / (res.avg_ms / 1000.0)
                : 0.0;

        std::cout
            << blocks << ","
            << bytes << ","
            << res.avg_ms << ","
            << res.p50_ms << ","
            << res.p99_ms << ","
            << avg_us_per_block << ","
            << p50_us_per_block << ","
            << p99_us_per_block << ","
            << effective_gbps
            << "\n";
    }

    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFreeHost(h_pool));

    return 0;
}
