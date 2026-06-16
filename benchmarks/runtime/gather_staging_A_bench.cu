#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
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

constexpr int BLOCK_COUNTS[] = {
    16, 64, 256, 1024
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double avg(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
}

// GPU scatter 커널: 연속 staging 버퍼의 i번째 block을
// 흩어진 목적지 d_pool의 dst_block_idx[i] 위치로 흩어 쓴다.
__global__ void scatter_blocks_kernel(
    float* __restrict__ d_pool,
    const float* __restrict__ d_staging,
    const int* __restrict__ dst_block_idx,
    int num_blocks,
    int block_elems
) {
    int i = blockIdx.x;
    if (i >= num_blocks) return;

    int dst_block = dst_block_idx[i];
    const float* src = d_staging + static_cast<size_t>(i) * block_elems;
    float* dst = d_pool + static_cast<size_t>(dst_block) * block_elems;

    for (int e = threadIdx.x; e < block_elems; e += blockDim.x) {
        dst[e] = src[e];
    }
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 500;
    int warmup = 50;
    unsigned int seed = 1234;

    if (argc >= 2) iterations = std::atoi(argv[1]);
    if (argc >= 3) warmup = std::atoi(argv[2]);
    if (argc >= 4) seed = static_cast<unsigned int>(std::atoi(argv[3]));

    const size_t block_elems =
        static_cast<size_t>(KV_BLOCK_SIZE) *
        static_cast<size_t>(GPT2_D_MODEL) *
        static_cast<size_t>(KV_PAIR);
    const size_t block_bytes = block_elems * sizeof(float);

    int max_blocks = 0;
    for (int b : BLOCK_COUNTS) max_blocks = std::max(max_blocks, b);

    const int POOL_MULT = 4;
    const int pool_blocks = max_blocks * POOL_MULT;

    const size_t pool_bytes =
        static_cast<size_t>(pool_blocks) * block_bytes;
    const size_t contig_bytes =
        static_cast<size_t>(max_blocks) * block_bytes;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "max_blocks=" << max_blocks << "\n";
    std::cout << "pool_blocks=" << pool_blocks << " (mult=" << POOL_MULT << ")\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "NOTE: CPU gather removed. host assumed already contiguous "
                 "(swap_out stored contiguously).\n\n";

    // host 버퍼.
    //  h_scattered : baseline 용. victim 데이터가 pool 레이아웃대로 흩어져 저장.
    //  h_contig    : staging 용. swap_out이 이미 연속으로 저장해 둔 버퍼.
    float* h_scattered = nullptr;
    float* h_contig = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_scattered), pool_bytes));
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_contig), contig_bytes));

    for (size_t i = 0; i < pool_bytes / sizeof(float); ++i) {
        h_scattered[i] = static_cast<float>(i % 1024);
    }
    for (size_t i = 0; i < contig_bytes / sizeof(float); ++i) {
        h_contig[i] = static_cast<float>(i % 1024);
    }

    // device 버퍼.
    float* d_pool = nullptr;
    float* d_staging = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pool), pool_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_staging), contig_bytes));

    int* d_dst_idx = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst_idx),
        static_cast<size_t>(max_blocks) * sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_a, ev_b;
    CUDA_CHECK(cudaEventCreate(&ev_a));
    CUDA_CHECK(cudaEventCreate(&ev_b));

    std::mt19937 rng(seed);

    std::cout
        << "method,blocks,bytes,"
        << "base_h2d_ms,base_bandwidth_GBps,"
        << "stg_h2d_ms,stg_scatter_ms,stg_total_ms,stg_bandwidth_GBps,"
        << "speedup_total,h2d_speedup,"
        << "base_p99_ms,stg_p99_ms\n";

    for (int blocks : BLOCK_COUNTS) {
        const size_t bytes =
            static_cast<size_t>(blocks) * block_bytes;

        // 흩어진 목적지 인덱스 생성.
        std::vector<int> pool_perm(pool_blocks);
        std::iota(pool_perm.begin(), pool_perm.end(), 0);
        std::shuffle(pool_perm.begin(), pool_perm.end(), rng);

        std::vector<int> dst_idx(blocks);
        for (int i = 0; i < blocks; ++i) dst_idx[i] = pool_perm[i];

        CUDA_CHECK(cudaMemcpy(
            d_dst_idx, dst_idx.data(),
            static_cast<size_t>(blocks) * sizeof(int),
            cudaMemcpyHostToDevice));

        int threads = 256;
        int grid = blocks;

        // baseline: host 흩어진 위치 -> GPU 흩어진 위치, H2D N번.
        auto run_baseline = [&](double& h2d_ms) {
            CUDA_CHECK(cudaEventRecord(ev_a, stream));
            for (int i = 0; i < blocks; ++i) {
                int blk = dst_idx[i];
                size_t off = static_cast<size_t>(blk) * block_elems;
                CUDA_CHECK(cudaMemcpyAsync(
                    d_pool + off,
                    h_scattered + off,
                    block_bytes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_a, ev_b));
            h2d_ms = static_cast<double>(ms);
        };

        // staging(A): host 연속 -> 연속 H2D 1번 -> GPU scatter.
        // (CPU gather 없음)
        auto run_staging = [&](double& h2d_ms, double& scatter_ms,
                               double& total_ms) {
            // 연속 H2D
            CUDA_CHECK(cudaEventRecord(ev_a, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_staging, h_contig, bytes,
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));
            float h2d = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&h2d, ev_a, ev_b));

            // GPU scatter
            CUDA_CHECK(cudaEventRecord(ev_a, stream));
            scatter_blocks_kernel<<<grid, threads, 0, stream>>>(
                d_pool, d_staging, d_dst_idx, blocks,
                static_cast<int>(block_elems));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));
            float scat = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&scat, ev_a, ev_b));

            h2d_ms = static_cast<double>(h2d);
            scatter_ms = static_cast<double>(scat);
            total_ms = h2d_ms + scatter_ms;
        };

        double t1, t2, t3;
        for (int i = 0; i < warmup; ++i) {
            run_baseline(t1);
            run_staging(t1, t2, t3);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> base_h2d;
        std::vector<double> stg_h2d, stg_scat, stg_total;

        for (int i = 0; i < iterations; ++i) {
            double h2d;
            run_baseline(h2d);
            base_h2d.push_back(h2d);

            double sh2d, scat, stot;
            run_staging(sh2d, scat, stot);
            stg_h2d.push_back(sh2d);
            stg_scat.push_back(scat);
            stg_total.push_back(stot);
        }

        double base_h2d_avg = avg(base_h2d);
        double stg_h2d_avg = avg(stg_h2d);
        double stg_scat_avg = avg(stg_scat);
        double stg_total_avg = avg(stg_total);

        double gb = static_cast<double>(bytes) / 1e9;
        double base_bw = (base_h2d_avg > 0.0)
            ? gb / (base_h2d_avg / 1000.0) : 0.0;
        double stg_bw = (stg_h2d_avg > 0.0)
            ? gb / (stg_h2d_avg / 1000.0) : 0.0;

        double speedup_total = (stg_total_avg > 0.0)
            ? base_h2d_avg / stg_total_avg : 0.0;
        double h2d_speedup = (stg_h2d_avg > 0.0)
            ? base_h2d_avg / stg_h2d_avg : 0.0;

        double base_p99 = percentile(base_h2d, 99.0);
        double stg_p99 = percentile(stg_total, 99.0);

        std::cout
            << "gather_staging_A,"
            << blocks << ","
            << bytes << ","
            << base_h2d_avg << ","
            << base_bw << ","
            << stg_h2d_avg << ","
            << stg_scat_avg << ","
            << stg_total_avg << ","
            << stg_bw << ","
            << speedup_total << ","
            << h2d_speedup << ","
            << base_p99 << ","
            << stg_p99
            << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(ev_a));
    CUDA_CHECK(cudaEventDestroy(ev_b));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dst_idx));
    CUDA_CHECK(cudaFree(d_staging));
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFreeHost(h_contig));
    CUDA_CHECK(cudaFreeHost(h_scattered));

    return 0;
}
