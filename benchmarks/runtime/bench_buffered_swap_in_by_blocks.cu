#include "constants.h"

#include <cuda_runtime.h>

#include <algorithm>
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

struct Sample {
    float h2d_ms = 0.0f;
    float scatter_ms = 0.0f;
    float total_ms = 0.0f;
};

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

__global__ void scatter_from_buffer_to_blocks_kernel(
    float* __restrict__ d_blocks,
    const float* __restrict__ d_buffer,
    const int* __restrict__ d_dst_blocks,
    int num_blocks,
    int block_elems
) {
    int block_i = blockIdx.x;

    if (block_i >= num_blocks) {
        return;
    }

    int dst_block = d_dst_blocks[block_i];

    const float* src =
        d_buffer + static_cast<size_t>(block_i) * block_elems;

    float* dst =
        d_blocks + static_cast<size_t>(dst_block) * block_elems;

    int vec4_elems = block_elems / 4;
    int tail_start = vec4_elems * 4;

    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    for (int v = threadIdx.x; v < vec4_elems; v += blockDim.x) {
        dst4[v] = src4[v];
    }

    for (int e = tail_start + threadIdx.x; e < block_elems; e += blockDim.x) {
        dst[e] = src[e];
    }
}

Sample run_buffered_swap_in_once(
    float* d_blocks,
    float* d_buffer,
    const float* h_contiguous_blocks,
    const int* d_dst_blocks,
    int blocks,
    size_t block_elems,
    size_t bytes,
    cudaStream_t stream
) {
    cudaEvent_t start;
    cudaEvent_t h2d_done;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&h2d_done));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_buffer,
        h_contiguous_blocks,
        bytes,
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaEventRecord(h2d_done, stream));

    constexpr int threads = 256;
    int grid = blocks;

    scatter_from_buffer_to_blocks_kernel<<<grid, threads, 0, stream>>>(
        d_blocks,
        d_buffer,
        d_dst_blocks,
        blocks,
        static_cast<int>(block_elems)
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    Sample sample;

    CUDA_CHECK(cudaEventElapsedTime(&sample.h2d_ms, start, h2d_done));
    CUDA_CHECK(cudaEventElapsedTime(&sample.scatter_ms, h2d_done, stop));
    CUDA_CHECK(cudaEventElapsedTime(&sample.total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(h2d_done));
    CUDA_CHECK(cudaEventDestroy(stop));

    return sample;
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
            << "usage: ./bench_buffered_swap_in_by_blocks "
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

    const size_t max_bytes =
        static_cast<size_t>(max_blocks) * block_bytes;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
    std::cout << "concurrentKernels=" << prop.concurrentKernels << "\n";
    std::cout << "block_elems=" << block_elems << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "NOTE: buffered swap_in uses one large H2D copy into GPU buffer.\n";
    std::cout << "NOTE: then a GPU scatter kernel copies buffer blocks into destination blocks.\n";
    std::cout << "NOTE: no BlockManager, no RealKvAllocator.\n\n";

    float* h_contiguous_blocks = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_contiguous_blocks),
        max_bytes
    ));

    fill_host_blocks(
        h_contiguous_blocks,
        max_blocks,
        block_elems
    );

    float* d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer),
        max_bytes
    ));

    float* d_blocks = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_blocks),
        max_bytes
    ));

    int* d_dst_blocks = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst_blocks),
        static_cast<size_t>(max_blocks) * sizeof(int)
    ));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking
    ));

    std::mt19937 rng(seed);

    std::cout
        << "blocks,bytes,"
        << "total_avg_ms,total_p50_ms,total_p99_ms,"
        << "h2d_avg_ms,scatter_avg_ms,"
        << "avg_us_per_block,effective_gbps\n";

    for (int blocks : BLOCK_COUNTS) {
        size_t bytes = static_cast<size_t>(blocks) * block_bytes;

        std::vector<int> dst_blocks =
            make_destination_blocks(blocks, rng);

        CUDA_CHECK(cudaMemcpy(
            d_dst_blocks,
            dst_blocks.data(),
            static_cast<size_t>(blocks) * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        for (int i = 0; i < warmup; ++i) {
            (void)run_buffered_swap_in_once(
                d_blocks,
                d_buffer,
                h_contiguous_blocks,
                d_dst_blocks,
                blocks,
                block_elems,
                bytes,
                stream
            );
        }

        std::vector<double> total_samples;
        std::vector<double> h2d_samples;
        std::vector<double> scatter_samples;

        total_samples.reserve(iterations);
        h2d_samples.reserve(iterations);
        scatter_samples.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            Sample sample = run_buffered_swap_in_once(
                d_blocks,
                d_buffer,
                h_contiguous_blocks,
                d_dst_blocks,
                blocks,
                block_elems,
                bytes,
                stream
            );

            total_samples.push_back(sample.total_ms);
            h2d_samples.push_back(sample.h2d_ms);
            scatter_samples.push_back(sample.scatter_ms);
        }

        Result total_res = summarize(total_samples);
        Result h2d_res = summarize(h2d_samples);
        Result scatter_res = summarize(scatter_samples);

        double avg_us_per_block =
            (total_res.avg_ms * 1000.0) / static_cast<double>(blocks);

        double effective_gbps =
            total_res.avg_ms > 0.0
                ? (static_cast<double>(bytes) / 1.0e9) /
                  (total_res.avg_ms / 1000.0)
                : 0.0;

        std::cout
            << blocks << ","
            << bytes << ","
            << total_res.avg_ms << ","
            << total_res.p50_ms << ","
            << total_res.p99_ms << ","
            << h2d_res.avg_ms << ","
            << scatter_res.avg_ms << ","
            << avg_us_per_block << ","
            << effective_gbps
            << "\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_dst_blocks));
    CUDA_CHECK(cudaFree(d_blocks));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFreeHost(h_contiguous_blocks));

    return 0;
}