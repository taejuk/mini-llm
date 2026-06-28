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

struct DirectSample {
    double wall_total_ms = 0.0;
    float event_total_ms = 0.0f;
};

struct BufferedSample {
    double wall_total_ms = 0.0;
    float event_total_ms = 0.0f;
    float event_h2d_ms = 0.0f;
    float event_scatter_ms = 0.0f;
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

DirectSample run_direct_async_once(
    float* d_blocks,
    const float* h_blocks,
    const std::vector<int>& dst_blocks,
    int blocks,
    size_t block_elems,
    size_t block_bytes,
    cudaStream_t stream,
    cudaEvent_t event_start,
    cudaEvent_t event_stop
) {
    CUDA_CHECK(cudaDeviceSynchronize());

    double wall_t0 = now_ms();

    CUDA_CHECK(cudaEventRecord(event_start, stream));

    for (int i = 0; i < blocks; ++i) {
        const float* src =
            h_blocks + static_cast<size_t>(i) * block_elems;

        float* dst =
            d_blocks + static_cast<size_t>(dst_blocks[i]) * block_elems;

        CUDA_CHECK(cudaMemcpyAsync(
            dst,
            src,
            block_bytes,
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    CUDA_CHECK(cudaEventRecord(event_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double wall_t1 = now_ms();

    DirectSample sample;
    sample.wall_total_ms = wall_t1 - wall_t0;

    CUDA_CHECK(cudaEventElapsedTime(
        &sample.event_total_ms,
        event_start,
        event_stop
    ));

    return sample;
}

BufferedSample run_buffered_async_once(
    float* d_blocks,
    float* d_buffer,
    const float* h_contiguous_blocks,
    const int* d_dst_blocks,
    int blocks,
    size_t block_elems,
    size_t bytes,
    cudaStream_t stream,
    cudaEvent_t event_start,
    cudaEvent_t event_h2d_done,
    cudaEvent_t event_stop
) {
    CUDA_CHECK(cudaDeviceSynchronize());

    double wall_t0 = now_ms();

    CUDA_CHECK(cudaEventRecord(event_start, stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_buffer,
        h_contiguous_blocks,
        bytes,
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaEventRecord(event_h2d_done, stream));

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

    CUDA_CHECK(cudaEventRecord(event_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double wall_t1 = now_ms();

    BufferedSample sample;
    sample.wall_total_ms = wall_t1 - wall_t0;

    CUDA_CHECK(cudaEventElapsedTime(
        &sample.event_total_ms,
        event_start,
        event_stop
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &sample.event_h2d_ms,
        event_start,
        event_h2d_done
    ));

    CUDA_CHECK(cudaEventElapsedTime(
        &sample.event_scatter_ms,
        event_h2d_done,
        event_stop
    ));

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
            << "usage: ./bench_fair_direct_vs_buffered_swap_in "
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
    std::cout << "NOTE: direct uses N cudaMemcpyAsync H2D calls, one per block.\n";
    std::cout << "NOTE: buffered uses one cudaMemcpyAsync H2D call plus one scatter kernel.\n";
    std::cout << "NOTE: both use pinned host memory, the same stream, wall time, and CUDA events.\n";
    std::cout << "NOTE: no BlockManager, no RealKvAllocator.\n\n";

    float* h_blocks = nullptr;

    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_blocks),
        max_bytes
    ));

    fill_host_blocks(
        h_blocks,
        max_blocks,
        block_elems
    );

    float* d_direct_blocks = nullptr;
    float* d_buffered_blocks = nullptr;
    float* d_buffer = nullptr;
    int* d_dst_blocks = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_direct_blocks),
        max_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffered_blocks),
        max_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer),
        max_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst_blocks),
        static_cast<size_t>(max_blocks) * sizeof(int)
    ));

    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking
    ));

    cudaEvent_t event_start;
    cudaEvent_t event_mid;
    cudaEvent_t event_stop;

    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_mid));
    CUDA_CHECK(cudaEventCreate(&event_stop));

    std::mt19937 rng(seed);

    std::cout
        << "blocks,bytes,"
        << "direct_wall_avg_ms,direct_wall_p50_ms,direct_wall_p99_ms,"
        << "direct_event_avg_ms,direct_event_p50_ms,direct_event_p99_ms,"
        << "direct_wall_us_per_block,direct_event_us_per_block,"
        << "direct_wall_gbps,direct_event_gbps,"
        << "buffered_wall_avg_ms,buffered_wall_p50_ms,buffered_wall_p99_ms,"
        << "buffered_event_avg_ms,buffered_event_p50_ms,buffered_event_p99_ms,"
        << "buffered_h2d_avg_ms,buffered_scatter_avg_ms,"
        << "buffered_wall_us_per_block,buffered_event_us_per_block,"
        << "buffered_wall_gbps,buffered_event_gbps,"
        << "wall_speedup,event_speedup\n";

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
            (void)run_direct_async_once(
                d_direct_blocks,
                h_blocks,
                dst_blocks,
                blocks,
                block_elems,
                block_bytes,
                stream,
                event_start,
                event_stop
            );

            (void)run_buffered_async_once(
                d_buffered_blocks,
                d_buffer,
                h_blocks,
                d_dst_blocks,
                blocks,
                block_elems,
                bytes,
                stream,
                event_start,
                event_mid,
                event_stop
            );
        }

        std::vector<double> direct_wall_samples;
        std::vector<double> direct_event_samples;

        std::vector<double> buffered_wall_samples;
        std::vector<double> buffered_event_samples;
        std::vector<double> buffered_h2d_samples;
        std::vector<double> buffered_scatter_samples;

        direct_wall_samples.reserve(iterations);
        direct_event_samples.reserve(iterations);

        buffered_wall_samples.reserve(iterations);
        buffered_event_samples.reserve(iterations);
        buffered_h2d_samples.reserve(iterations);
        buffered_scatter_samples.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            DirectSample direct_sample = run_direct_async_once(
                d_direct_blocks,
                h_blocks,
                dst_blocks,
                blocks,
                block_elems,
                block_bytes,
                stream,
                event_start,
                event_stop
            );

            BufferedSample buffered_sample = run_buffered_async_once(
                d_buffered_blocks,
                d_buffer,
                h_blocks,
                d_dst_blocks,
                blocks,
                block_elems,
                bytes,
                stream,
                event_start,
                event_mid,
                event_stop
            );

            direct_wall_samples.push_back(direct_sample.wall_total_ms);
            direct_event_samples.push_back(direct_sample.event_total_ms);

            buffered_wall_samples.push_back(buffered_sample.wall_total_ms);
            buffered_event_samples.push_back(buffered_sample.event_total_ms);
            buffered_h2d_samples.push_back(buffered_sample.event_h2d_ms);
            buffered_scatter_samples.push_back(buffered_sample.event_scatter_ms);
        }

        Result direct_wall_res = summarize(direct_wall_samples);
        Result direct_event_res = summarize(direct_event_samples);

        Result buffered_wall_res = summarize(buffered_wall_samples);
        Result buffered_event_res = summarize(buffered_event_samples);
        Result buffered_h2d_res = summarize(buffered_h2d_samples);
        Result buffered_scatter_res = summarize(buffered_scatter_samples);

        double direct_wall_us_per_block =
            (direct_wall_res.avg_ms * 1000.0) /
            static_cast<double>(blocks);

        double direct_event_us_per_block =
            (direct_event_res.avg_ms * 1000.0) /
            static_cast<double>(blocks);

        double buffered_wall_us_per_block =
            (buffered_wall_res.avg_ms * 1000.0) /
            static_cast<double>(blocks);

        double buffered_event_us_per_block =
            (buffered_event_res.avg_ms * 1000.0) /
            static_cast<double>(blocks);

        double direct_wall_gbps =
            direct_wall_res.avg_ms > 0.0
                ? (static_cast<double>(bytes) / 1.0e9) /
                  (direct_wall_res.avg_ms / 1000.0)
                : 0.0;

        double direct_event_gbps =
            direct_event_res.avg_ms > 0.0
                ? (static_cast<double>(bytes) / 1.0e9) /
                  (direct_event_res.avg_ms / 1000.0)
                : 0.0;

        double buffered_wall_gbps =
            buffered_wall_res.avg_ms > 0.0
                ? (static_cast<double>(bytes) / 1.0e9) /
                  (buffered_wall_res.avg_ms / 1000.0)
                : 0.0;

        double buffered_event_gbps =
            buffered_event_res.avg_ms > 0.0
                ? (static_cast<double>(bytes) / 1.0e9) /
                  (buffered_event_res.avg_ms / 1000.0)
                : 0.0;

        double wall_speedup =
            buffered_wall_res.avg_ms > 0.0
                ? direct_wall_res.avg_ms / buffered_wall_res.avg_ms
                : 0.0;

        double event_speedup =
            buffered_event_res.avg_ms > 0.0
                ? direct_event_res.avg_ms / buffered_event_res.avg_ms
                : 0.0;

        std::cout
            << blocks << ","
            << bytes << ","

            << direct_wall_res.avg_ms << ","
            << direct_wall_res.p50_ms << ","
            << direct_wall_res.p99_ms << ","

            << direct_event_res.avg_ms << ","
            << direct_event_res.p50_ms << ","
            << direct_event_res.p99_ms << ","

            << direct_wall_us_per_block << ","
            << direct_event_us_per_block << ","

            << direct_wall_gbps << ","
            << direct_event_gbps << ","

            << buffered_wall_res.avg_ms << ","
            << buffered_wall_res.p50_ms << ","
            << buffered_wall_res.p99_ms << ","

            << buffered_event_res.avg_ms << ","
            << buffered_event_res.p50_ms << ","
            << buffered_event_res.p99_ms << ","

            << buffered_h2d_res.avg_ms << ","
            << buffered_scatter_res.avg_ms << ","

            << buffered_wall_us_per_block << ","
            << buffered_event_us_per_block << ","

            << buffered_wall_gbps << ","
            << buffered_event_gbps << ","

            << wall_speedup << ","
            << event_speedup
            << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(event_stop));
    CUDA_CHECK(cudaEventDestroy(event_mid));
    CUDA_CHECK(cudaEventDestroy(event_start));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_dst_blocks));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_buffered_blocks));
    CUDA_CHECK(cudaFree(d_direct_blocks));
    CUDA_CHECK(cudaFreeHost(h_blocks));

    return 0;
}