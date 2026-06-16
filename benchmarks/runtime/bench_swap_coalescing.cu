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

// 측정할 block 개수 (전체 전송량). 2의 거듭제곱으로 훑는다.
constexpr int BLOCK_COUNTS[] = {
    16, 64, 256, 1024
};

// batch 수 = memcpy 호출 횟수.
//   1        : 전체 block을 한 번의 큰 복사로 (합치기, 호출 최소)
//   blocks   : block당 한 번씩 (호출 최대, 원래 방식)
// 아래 값들 중 blocks보다 큰 것은 실행 시 blocks로 클램프된다.
constexpr int BATCH_COUNTS[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 1024
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

// 전체 blocks개를 batch_count개의 memcpy 호출로 나눠 H2D 전송한다.
// 각 호출은 연속된 여러 block을 한 번에 보낸다(연속 메모리 전제).
//
// 측정값:
//   wall_ms : cudaMemcpyAsync 호출들 + 동기화까지 포함한 CPU wall-clock
//             (호출 오버헤드 + 전송이 섞임)
//   gpu_ms  : cudaEvent로 잰 순수 GPU 전송 구간 시간 (호출 오버헤드 제외)
void run_batched_h2d(
    float* d_dst,
    float* h_src,
    int blocks,
    int batch_count,
    size_t block_elems,
    cudaStream_t stream,
    cudaEvent_t start_ev,
    cudaEvent_t stop_ev,
    double& wall_ms,
    double& gpu_ms
) {
    // blocks를 batch_count개로 최대한 균등 분할.
    // 나머지는 앞쪽 배치들에 1개씩 더 배분한다.
    int base = blocks / batch_count;
    int rem = blocks % batch_count;

    auto wall_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventRecord(start_ev, stream));

    int block_offset = 0;
    for (int i = 0; i < batch_count; ++i) {
        int this_blocks = base + (i < rem ? 1 : 0);
        if (this_blocks == 0) continue;

        size_t elem_offset =
            static_cast<size_t>(block_offset) * block_elems;
        size_t copy_bytes =
            static_cast<size_t>(this_blocks) * block_elems * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(
            d_dst + elem_offset,
            h_src + elem_offset,
            copy_bytes,
            cudaMemcpyHostToDevice,
            stream
        ));

        block_offset += this_blocks;
    }

    CUDA_CHECK(cudaEventRecord(stop_ev, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    auto wall_end = std::chrono::high_resolution_clock::now();

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));
    gpu_ms = static_cast<double>(elapsed_ms);

    wall_ms = std::chrono::duration<double, std::milli>(
        wall_end - wall_start
    ).count();
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 1000;
    int warmup = 100;

    if (argc >= 2) iterations = std::atoi(argv[1]);
    if (argc >= 3) warmup = std::atoi(argv[2]);

    const size_t block_elems =
        static_cast<size_t>(KV_BLOCK_SIZE) *
        static_cast<size_t>(GPT2_D_MODEL) *
        static_cast<size_t>(KV_PAIR);
    const size_t block_bytes = block_elems * sizeof(float);

    // 가장 큰 block 개수 기준으로 버퍼를 한 번 잡는다.
    int max_blocks = 0;
    for (int b : BLOCK_COUNTS) max_blocks = std::max(max_blocks, b);

    const size_t total_bytes =
        static_cast<size_t>(max_blocks) * block_bytes;
    const size_t total_elems = total_bytes / sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "max_blocks=" << max_blocks << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n\n";

    float* h_src = nullptr;
    float* d_dst = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_src), total_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dst), total_bytes));

    for (size_t i = 0; i < total_elems; ++i) {
        h_src[i] = static_cast<float>(i % 1024);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    std::cout
        << "direction,blocks,batch_count,copy_calls,blocks_per_call,bytes,"
        << "wall_avg_ms,wall_p50_ms,wall_p99_ms,"
        << "gpu_avg_ms,gpu_p50_ms,gpu_p99_ms,"
        << "call_overhead_avg_ms,call_overhead_ratio,"
        << "gpu_bandwidth_GBps\n";

    for (int blocks : BLOCK_COUNTS) {
        const size_t bytes =
            static_cast<size_t>(blocks) * block_bytes;

        // 이 blocks 값에서 이미 측정한 batch_count를 추적해 중복을 막는다.
        // (예: blocks=16일 때 batch 32/64/...는 모두 16으로 클램프되므로
        //  16은 한 번만 측정)
        int prev_batch = -1;

        for (int batch_req : BATCH_COUNTS) {
            // 배치 수는 block 개수를 넘을 수 없다.
            int batch_count = std::min(batch_req, blocks);

            if (batch_count == prev_batch) continue;
            prev_batch = batch_count;

            double wall_ms = 0.0, gpu_ms = 0.0;

            for (int i = 0; i < warmup; ++i) {
                run_batched_h2d(
                    d_dst, h_src, blocks, batch_count,
                    block_elems, stream, start_ev, stop_ev,
                    wall_ms, gpu_ms
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> wall_times, gpu_times;
            wall_times.reserve(iterations);
            gpu_times.reserve(iterations);

            for (int i = 0; i < iterations; ++i) {
                run_batched_h2d(
                    d_dst, h_src, blocks, batch_count,
                    block_elems, stream, start_ev, stop_ev,
                    wall_ms, gpu_ms
                );
                wall_times.push_back(wall_ms);
                gpu_times.push_back(gpu_ms);
            }

            double wall_avg =
                std::accumulate(wall_times.begin(), wall_times.end(), 0.0) /
                static_cast<double>(wall_times.size());
            double gpu_avg =
                std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) /
                static_cast<double>(gpu_times.size());

            double wall_p50 = percentile(wall_times, 50.0);
            double wall_p99 = percentile(wall_times, 99.0);
            double gpu_p50 = percentile(gpu_times, 50.0);
            double gpu_p99 = percentile(gpu_times, 99.0);

            // 호출 오버헤드 = wall - gpu. 호출 수가 많을수록 커지면
            // 병목이 CPU 측 디스패치(호출 횟수)라는 직접 증거.
            double call_overhead = wall_avg - gpu_avg;

            // 호출 오버헤드가 전체 wall에서 차지하는 비율.
            // 1에 가까울수록 "전송보다 호출이 비싸다"는 뜻.
            double call_overhead_ratio =
                (wall_avg > 0.0) ? (call_overhead / wall_avg) : 0.0;

            double gb = static_cast<double>(bytes) / 1e9;
            double gpu_sec = gpu_avg / 1000.0;
            double gpu_bw = (gpu_sec > 0.0) ? (gb / gpu_sec) : 0.0;

            int blocks_per_call = blocks / batch_count;

            std::cout
                << "H2D,"
                << blocks << ","
                << batch_count << ","
                << batch_count << ","
                << blocks_per_call << ","
                << bytes << ","
                << wall_avg << ","
                << wall_p50 << ","
                << wall_p99 << ","
                << gpu_avg << ","
                << gpu_p50 << ","
                << gpu_p99 << ","
                << call_overhead << ","
                << call_overhead_ratio << ","
                << gpu_bw
                << "\n";
        }
    }

    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_src));

    return 0;
}
