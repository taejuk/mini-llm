#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
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

// GPU scatter 커널: 연속 staging 버퍼(d_staging)의 block들을
// 흩어진 목적지(d_pool)의 dst_block_idx[] 위치로 흩어 쓴다.
// 한 블록(블록 인덱스 i)을 CUDA 블록 하나가 담당하고,
// 블록 내부 원소들을 스레드들이 나눠 복사한다.
__global__ void scatter_blocks_kernel(
    float* __restrict__ d_pool,        // 흩어진 목적지 (전체 pool)
    const float* __restrict__ d_staging, // 연속 staging
    const int* __restrict__ dst_block_idx, // staging i번째 -> pool의 어느 block
    int num_blocks,
    int block_elems
) {
    int i = blockIdx.x;  // staging 상의 i번째 block
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

    // pool은 흩어짐을 재현하기 위해 실제 옮길 block 수보다 크게 잡는다.
    // (victim block들이 큰 pool 안에 듬성듬성 박혀 있는 상황)
    const int POOL_MULT = 4;
    const int pool_blocks = max_blocks * POOL_MULT;

    const size_t pool_bytes =
        static_cast<size_t>(pool_blocks) * block_bytes;
    const size_t staging_bytes =
        static_cast<size_t>(max_blocks) * block_bytes;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "max_blocks=" << max_blocks << "\n";
    std::cout << "pool_blocks=" << pool_blocks << " (mult=" << POOL_MULT << ")\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n\n";

    // ---- host 메모리 ----
    // h_scattered_src: victim 데이터가 흩어져 있는 host 원본 (pinned).
    //   실제로는 swap-out 때 흩어져 저장됐다고 가정. 여기선 pool 레이아웃과
    //   동일하게 잡아, src도 흩어진 위치에서 읽는 상황을 재현한다.
    // h_contig: CPU gather 결과를 담는 연속 버퍼 (pinned, 큰 H2D용).
    float* h_scattered_src = nullptr;
    float* h_contig = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_scattered_src), pool_bytes));
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_contig), staging_bytes));

    for (size_t i = 0; i < pool_bytes / sizeof(float); ++i) {
        h_scattered_src[i] = static_cast<float>(i % 1024);
    }

    // ---- device 메모리 ----
    // d_pool: 흩어진 목적지 (전체 pool).
    // d_staging: 연속 staging 버퍼.
    float* d_pool = nullptr;
    float* d_staging = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pool), pool_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_staging), staging_bytes));

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
        // baseline: 흩어진 채 H2D N번
        << "base_h2d_ms,base_total_ms,base_bandwidth_GBps,"
        // staging: cpu_gather + 연속 H2D + gpu_scatter
        << "stg_cpu_gather_ms,stg_h2d_ms,stg_scatter_ms,stg_total_ms,"
        << "stg_bandwidth_GBps,"
        // 비교
        << "speedup_total,h2d_speedup\n";

    for (int blocks : BLOCK_COUNTS) {
        const size_t bytes =
            static_cast<size_t>(blocks) * block_bytes;

        // 이 blocks개를 pool_blocks 중 어디에 흩어 놓을지 인덱스를 정한다.
        // pool 인덱스를 섞어서 앞 blocks개를 victim의 흩어진 위치로 사용.
        std::vector<int> pool_perm(pool_blocks);
        std::iota(pool_perm.begin(), pool_perm.end(), 0);
        std::shuffle(pool_perm.begin(), pool_perm.end(), rng);

        std::vector<int> dst_idx(blocks);
        for (int i = 0; i < blocks; ++i) dst_idx[i] = pool_perm[i];

        CUDA_CHECK(cudaMemcpy(
            d_dst_idx, dst_idx.data(),
            static_cast<size_t>(blocks) * sizeof(int),
            cudaMemcpyHostToDevice));

        // scatter 커널 구성
        int threads = 256;
        int grid = blocks;  // block 하나당 CUDA block 하나

        // ---------- 측정 버퍼 ----------
        std::vector<double> base_h2d, base_total;
        std::vector<double> stg_cpu, stg_h2d, stg_scatter, stg_total;

        // 한 번의 baseline 실행: 흩어진 src -> 흩어진 dst, H2D N번
        auto run_baseline = [&](double& h2d_ms, double& total_ms) {
            auto t0 = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaEventRecord(ev_a, stream));
            for (int i = 0; i < blocks; ++i) {
                int dst_block = dst_idx[i];
                // src도 동일하게 흩어진 위치에서 읽는다(공정성).
                size_t src_off =
                    static_cast<size_t>(dst_block) * block_elems;
                size_t dst_off =
                    static_cast<size_t>(dst_block) * block_elems;

                CUDA_CHECK(cudaMemcpyAsync(
                    d_pool + dst_off,
                    h_scattered_src + src_off,
                    block_bytes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            auto t1 = std::chrono::high_resolution_clock::now();

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_a, ev_b));
            h2d_ms = static_cast<double>(ms);
            total_ms = std::chrono::duration<double, std::milli>(
                t1 - t0).count();
        };

        // 한 번의 staging 실행: cpu gather -> 연속 H2D -> gpu scatter
        auto run_staging = [&](double& cpu_ms, double& h2d_ms,
                               double& scatter_ms, double& total_ms) {
            auto t0 = std::chrono::high_resolution_clock::now();

            // (1) CPU gather: 흩어진 host 조각 -> 연속 host 버퍼
            for (int i = 0; i < blocks; ++i) {
                int src_block = dst_idx[i];
                std::memcpy(
                    h_contig + static_cast<size_t>(i) * block_elems,
                    h_scattered_src +
                        static_cast<size_t>(src_block) * block_elems,
                    block_bytes);
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            // (2) 연속 H2D 한 번
            CUDA_CHECK(cudaEventRecord(ev_a, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_staging, h_contig, bytes,
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            float h2d = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&h2d, ev_a, ev_b));

            // (3) GPU scatter: 연속 staging -> 흩어진 d_pool
            CUDA_CHECK(cudaEventRecord(ev_a, stream));
            scatter_blocks_kernel<<<grid, threads, 0, stream>>>(
                d_pool, d_staging, d_dst_idx, blocks,
                static_cast<int>(block_elems));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            float scat = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&scat, ev_a, ev_b));

            auto t2 = std::chrono::high_resolution_clock::now();

            cpu_ms = std::chrono::duration<double, std::milli>(
                t1 - t0).count();
            h2d_ms = static_cast<double>(h2d);
            scatter_ms = static_cast<double>(scat);
            total_ms = std::chrono::duration<double, std::milli>(
                t2 - t0).count();
        };

        double a, b, c, d;
        for (int i = 0; i < warmup; ++i) {
            run_baseline(a, b);
            run_staging(a, b, c, d);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < iterations; ++i) {
            double h2d_ms, total_ms;
            run_baseline(h2d_ms, total_ms);
            base_h2d.push_back(h2d_ms);
            base_total.push_back(total_ms);

            double cpu_ms, sh2d_ms, scat_ms, stot_ms;
            run_staging(cpu_ms, sh2d_ms, scat_ms, stot_ms);
            stg_cpu.push_back(cpu_ms);
            stg_h2d.push_back(sh2d_ms);
            stg_scatter.push_back(scat_ms);
            stg_total.push_back(stot_ms);
        }

        double base_h2d_avg = avg(base_h2d);
        double base_total_avg = avg(base_total);
        double stg_cpu_avg = avg(stg_cpu);
        double stg_h2d_avg = avg(stg_h2d);
        double stg_scat_avg = avg(stg_scatter);
        double stg_total_avg = avg(stg_total);

        double gb = static_cast<double>(bytes) / 1e9;
        double base_bw = (base_h2d_avg > 0.0)
            ? gb / (base_h2d_avg / 1000.0) : 0.0;
        double stg_bw = (stg_h2d_avg > 0.0)
            ? gb / (stg_h2d_avg / 1000.0) : 0.0;

        // speedup: baseline 전체 대비 staging 전체 (>1이면 staging이 빠름)
        double speedup_total = (stg_total_avg > 0.0)
            ? base_total_avg / stg_total_avg : 0.0;
        // 순수 H2D 구간만의 개선 (PCIe 효율 이득)
        double h2d_speedup = (stg_h2d_avg > 0.0)
            ? base_h2d_avg / stg_h2d_avg : 0.0;

        std::cout
            << "gather_staging,"
            << blocks << ","
            << bytes << ","
            << base_h2d_avg << ","
            << base_total_avg << ","
            << base_bw << ","
            << stg_cpu_avg << ","
            << stg_h2d_avg << ","
            << stg_scat_avg << ","
            << stg_total_avg << ","
            << stg_bw << ","
            << speedup_total << ","
            << h2d_speedup
            << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(ev_a));
    CUDA_CHECK(cudaEventDestroy(ev_b));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dst_idx));
    CUDA_CHECK(cudaFree(d_staging));
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFreeHost(h_contig));
    CUDA_CHECK(cudaFreeHost(h_scattered_src));

    return 0;
}
