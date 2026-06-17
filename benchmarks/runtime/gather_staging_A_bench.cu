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

struct Sample {
    double h2d_ms = 0.0;
    double scatter_ms = 0.0;
    double total_ms = 0.0;
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

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

// scalar scatter:
// staging의 i번째 block을 d_pool의 dst_block_idx[i] block으로 복사한다.
__global__ void scatter_blocks_scalar_kernel(
    float* __restrict__ d_pool,
    const float* __restrict__ d_staging,
    const int* __restrict__ dst_block_idx,
    int num_blocks,
    int block_elems
) {
    int block_i = blockIdx.x;

    if (block_i >= num_blocks) {
        return;
    }

    int dst_block = dst_block_idx[block_i];

    const float* src =
        d_staging + static_cast<size_t>(block_i) * block_elems;

    float* dst =
        d_pool + static_cast<size_t>(dst_block) * block_elems;

    for (int e = threadIdx.x; e < block_elems; e += blockDim.x) {
        dst[e] = src[e];
    }
}

// vectorized scatter:
// staging의 i번째 block을 d_pool의 dst_block_idx[i] block으로 복사한다.
// float4 단위로 복사해서 scalar 대입보다 instruction 수를 줄인다.
__global__ void scatter_blocks_vec4_kernel(
    float* __restrict__ d_pool,
    const float* __restrict__ d_staging,
    const int* __restrict__ dst_block_idx,
    int num_blocks,
    int block_elems
) {
    int block_i = blockIdx.x;

    if (block_i >= num_blocks) {
        return;
    }

    int dst_block = dst_block_idx[block_i];

    const float* src =
        d_staging + static_cast<size_t>(block_i) * block_elems;

    float* dst =
        d_pool + static_cast<size_t>(dst_block) * block_elems;

    int vec4_elems = block_elems / 4;
    int tail_start = vec4_elems * 4;

    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);

    for (int v = threadIdx.x; v < vec4_elems; v += blockDim.x) {
        dst4[v] = src4[v];
    }

    // 일반성 유지용 tail 처리.
    // 현재 block_elems = 16 * 768 * 2 = 24576 이라 4로 나누어 떨어져서 실행되지 않는다.
    for (int e = tail_start + threadIdx.x; e < block_elems; e += blockDim.x) {
        dst[e] = src[e];
    }
}

void print_row(
    const char* method,
    int blocks,
    size_t bytes,
    const std::vector<Sample>& samples,
    double plain_total_avg
) {
    std::vector<double> h2d;
    std::vector<double> scatter;
    std::vector<double> total;

    h2d.reserve(samples.size());
    scatter.reserve(samples.size());
    total.reserve(samples.size());

    for (const Sample& s : samples) {
        h2d.push_back(s.h2d_ms);
        scatter.push_back(s.scatter_ms);
        total.push_back(s.total_ms);
    }

    double h2d_avg = avg(h2d);
    double scatter_avg = avg(scatter);
    double total_avg = avg(total);

    double gb = static_cast<double>(bytes) / 1e9;

    double h2d_bw = (h2d_avg > 0.0)
        ? gb / (h2d_avg / 1000.0)
        : 0.0;

    double e2e_bw = (total_avg > 0.0)
        ? gb / (total_avg / 1000.0)
        : 0.0;

    double speedup_vs_plain = (total_avg > 0.0)
        ? plain_total_avg / total_avg
        : 0.0;

    std::cout
        << method << ","
        << blocks << ","
        << bytes << ","
        << h2d_avg << ","
        << scatter_avg << ","
        << total_avg << ","
        << h2d_bw << ","
        << e2e_bw << ","
        << speedup_vs_plain << ","
        << percentile(total, 50.0) << ","
        << percentile(total, 99.0) << ","
        << percentile(scatter, 99.0)
        << "\n";
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
    for (int b : BLOCK_COUNTS) {
        max_blocks = std::max(max_blocks, b);
    }

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
    std::cout << "block_elems=" << block_elems << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "max_blocks=" << max_blocks << "\n";
    std::cout << "pool_blocks=" << pool_blocks << " (mult=" << POOL_MULT << ")\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "NOTE: host KV blocks are assumed contiguous, as in swap_out.\n";
    std::cout << "NOTE: vec4 scatter is valid because cudaMalloc is aligned and "
                 "block_elems is divisible by 4.\n\n";

    // CPU contiguous swap storage.
    float* h_contig = nullptr;
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_contig),
        contig_bytes
    ));

    for (size_t i = 0; i < contig_bytes / sizeof(float); ++i) {
        h_contig[i] = static_cast<float>(i % 1024);
    }

    // GPU pool and GPU staging buffer.
    float* d_pool = nullptr;
    float* d_staging = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_pool),
        pool_bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_staging),
        contig_bytes
    ));

    int* d_dst_idx = nullptr;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst_idx),
        static_cast<size_t>(max_blocks) * sizeof(int)
    ));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_a;
    cudaEvent_t ev_b;

    CUDA_CHECK(cudaEventCreate(&ev_a));
    CUDA_CHECK(cudaEventCreate(&ev_b));

    std::mt19937 rng(seed);

    std::cout
        << "method,blocks,bytes,"
        << "h2d_avg_ms,scatter_avg_ms,total_avg_ms,"
        << "h2d_bandwidth_GBps,e2e_bandwidth_GBps,"
        << "speedup_vs_plain,total_p50_ms,total_p99_ms,scatter_p99_ms\n";

    for (int blocks : BLOCK_COUNTS) {
        const size_t bytes =
            static_cast<size_t>(blocks) * block_bytes;

        // scattered GPU destination block ids.
        std::vector<int> pool_perm(pool_blocks);
        std::iota(pool_perm.begin(), pool_perm.end(), 0);
        std::shuffle(pool_perm.begin(), pool_perm.end(), rng);

        std::vector<int> dst_idx(blocks);

        for (int i = 0; i < blocks; ++i) {
            dst_idx[i] = pool_perm[i];
        }

        CUDA_CHECK(cudaMemcpy(
            d_dst_idx,
            dst_idx.data(),
            static_cast<size_t>(blocks) * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        constexpr int threads = 256;
        int grid = blocks;

        auto run_plain = [&]() -> Sample {
            // plain:
            // CPU contiguous block i -> GPU scattered dst_idx[i]
            // cudaMemcpyAsync를 block 개수만큼 호출.
            CUDA_CHECK(cudaEventRecord(ev_a, stream));

            for (int i = 0; i < blocks; ++i) {
                int dst_block = dst_idx[i];

                const float* src =
                    h_contig + static_cast<size_t>(i) * block_elems;

                float* dst =
                    d_pool + static_cast<size_t>(dst_block) * block_elems;

                CUDA_CHECK(cudaMemcpyAsync(
                    dst,
                    src,
                    block_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ));
            }

            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            double h2d_ms = elapsed_ms(ev_a, ev_b);

            return Sample{
                h2d_ms,
                0.0,
                h2d_ms
            };
        };

        auto run_staging_scalar = [&]() -> Sample {
            // 1. CPU contiguous -> GPU staging H2D 1번
            CUDA_CHECK(cudaEventRecord(ev_a, stream));

            CUDA_CHECK(cudaMemcpyAsync(
                d_staging,
                h_contig,
                bytes,
                cudaMemcpyHostToDevice,
                stream
            ));

            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            double h2d_ms = elapsed_ms(ev_a, ev_b);

            // 2. scalar GPU scatter
            CUDA_CHECK(cudaEventRecord(ev_a, stream));

            scatter_blocks_scalar_kernel<<<grid, threads, 0, stream>>>(
                d_pool,
                d_staging,
                d_dst_idx,
                blocks,
                static_cast<int>(block_elems)
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            double scatter_ms = elapsed_ms(ev_a, ev_b);

            return Sample{
                h2d_ms,
                scatter_ms,
                h2d_ms + scatter_ms
            };
        };

        auto run_staging_vec4 = [&]() -> Sample {
            // 1. CPU contiguous -> GPU staging H2D 1번
            CUDA_CHECK(cudaEventRecord(ev_a, stream));

            CUDA_CHECK(cudaMemcpyAsync(
                d_staging,
                h_contig,
                bytes,
                cudaMemcpyHostToDevice,
                stream
            ));

            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            double h2d_ms = elapsed_ms(ev_a, ev_b);

            // 2. vectorized GPU scatter
            CUDA_CHECK(cudaEventRecord(ev_a, stream));

            scatter_blocks_vec4_kernel<<<grid, threads, 0, stream>>>(
                d_pool,
                d_staging,
                d_dst_idx,
                blocks,
                static_cast<int>(block_elems)
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(ev_b, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_b));

            double scatter_ms = elapsed_ms(ev_a, ev_b);

            return Sample{
                h2d_ms,
                scatter_ms,
                h2d_ms + scatter_ms
            };
        };

        for (int i = 0; i < warmup; ++i) {
            (void)run_plain();
            (void)run_staging_scalar();
            (void)run_staging_vec4();
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<Sample> plain_samples;
        std::vector<Sample> scalar_samples;
        std::vector<Sample> vec4_samples;

        plain_samples.reserve(iterations);
        scalar_samples.reserve(iterations);
        vec4_samples.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            plain_samples.push_back(run_plain());
            scalar_samples.push_back(run_staging_scalar());
            vec4_samples.push_back(run_staging_vec4());
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> plain_total;
        plain_total.reserve(plain_samples.size());

        for (const Sample& s : plain_samples) {
            plain_total.push_back(s.total_ms);
        }

        double plain_total_avg = avg(plain_total);

        print_row(
            "plain_block_h2d",
            blocks,
            bytes,
            plain_samples,
            plain_total_avg
        );

        print_row(
            "staging_scalar",
            blocks,
            bytes,
            scalar_samples,
            plain_total_avg
        );

        print_row(
            "staging_vec4",
            blocks,
            bytes,
            vec4_samples,
            plain_total_avg
        );
    }

    CUDA_CHECK(cudaEventDestroy(ev_a));
    CUDA_CHECK(cudaEventDestroy(ev_b));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_dst_idx));
    CUDA_CHECK(cudaFree(d_staging));
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFreeHost(h_contig));

    return 0;
}