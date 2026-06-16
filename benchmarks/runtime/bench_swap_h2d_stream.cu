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

constexpr int STREAM_COUNTS[] = {
    1, 2, 4, 8, 16, 32, 128
};

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

// 한 번의 multi-stream 복사를 디스패치하고, 두 가지 시간을 돌려준다.
//   wall_ms  : cudaMemcpyAsync 호출 + 최종 동기화까지 포함한 CPU wall-clock
//              (호출 오버헤드 + 전송 시간이 섞여 있음)
//   gpu_ms   : cudaEvent로 잰 순수 GPU 전송 구간 시간
//              (CPU 호출 오버헤드가 빠진, 디바이스 타임라인 상의 복사 시간)
//
// multi-stream에서 "전체 복사 구간"을 cudaEvent로 재는 방법:
//   1) start 이벤트를 stream[0]에 기록한다.
//   2) 모든 copy stream이 start 이벤트를 기다리게(cudaStreamWaitEvent) 해서,
//      모든 복사가 동일한 start 시점 이후에 시작하도록 정렬한다.
//   3) 복사를 각 stream에 디스패치한다.
//   4) 모든 copy stream이 stream[0]에 합류하도록, 각 stream에 마커 이벤트를
//      찍고 stream[0]이 그 이벤트들을 기다리게 한다.
//   5) stop 이벤트를 stream[0]에 기록한다.
//   6) stop 이벤트까지 동기화한 뒤 start~stop 경과 시간을 읽는다.
void copy_blocks_h2d_async(
    float* d_dst,
    float* h_src,
    int blocks,
    size_t block_elems,
    size_t block_bytes,
    std::vector<cudaStream_t>& streams,
    int stream_count,
    cudaEvent_t start_ev,
    cudaEvent_t stop_ev,
    std::vector<cudaEvent_t>& join_evs,
    double& wall_ms,
    double& gpu_ms
) {
    int active_streams = std::min(stream_count, blocks);

    auto wall_start = std::chrono::high_resolution_clock::now();

    // (1) start 이벤트를 stream[0]에 기록
    CUDA_CHECK(cudaEventRecord(start_ev, streams[0]));

    // (2) 모든 active stream이 start 시점 이후에 시작하도록 정렬
    //     stream[0]은 이미 start를 기록했으므로 그 이후 작업은 자동 정렬됨.
    for (int sid = 1; sid < active_streams; ++sid) {
        CUDA_CHECK(cudaStreamWaitEvent(streams[sid], start_ev, 0));
    }

    // (3) 복사 디스패치 (block을 round-robin으로 stream에 분배)
    for (int b = 0; b < blocks; ++b) {
        int sid = b % active_streams;

        float* src = h_src + static_cast<size_t>(b) * block_elems;
        float* dst = d_dst + static_cast<size_t>(b) * block_elems;

        CUDA_CHECK(cudaMemcpyAsync(
            dst,
            src,
            block_bytes,
            cudaMemcpyHostToDevice,
            streams[sid]
        ));
    }

    // (4) 각 보조 stream이 stream[0]에 합류하도록 마커 이벤트 사용
    for (int sid = 1; sid < active_streams; ++sid) {
        CUDA_CHECK(cudaEventRecord(join_evs[sid], streams[sid]));
        CUDA_CHECK(cudaStreamWaitEvent(streams[0], join_evs[sid], 0));
    }

    // (5) stop 이벤트를 stream[0]에 기록 (모든 복사 완료 후 시점)
    CUDA_CHECK(cudaEventRecord(stop_ev, streams[0]));

    // (6) stop 이벤트까지 대기 후 순수 GPU 전송 시간 측정
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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device=" << prop.name << "\n";
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
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

    constexpr int MAX_STREAMS = 128;
    std::vector<cudaStream_t> streams(MAX_STREAMS);

    for (int i = 0; i < MAX_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 타이밍/합류용 이벤트.
    // 타이밍 이벤트(start/stop)는 기본 플래그(타이밍 활성).
    // 합류 마커 이벤트는 cudaEventDisableTiming으로 만들어 오버헤드를 줄인다.
    cudaEvent_t start_ev;
    cudaEvent_t stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    std::vector<cudaEvent_t> join_evs(MAX_STREAMS);
    for (int i = 0; i < MAX_STREAMS; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(
            &join_evs[i],
            cudaEventDisableTiming
        ));
    }

    std::cout
        << "direction,mode,blocks,requested_streams,active_streams,"
        << "copy_calls,bytes,"
        << "wall_avg_ms,wall_p50_ms,wall_p99_ms,"
        << "gpu_avg_ms,gpu_p50_ms,gpu_p99_ms,"
        << "call_overhead_avg_ms,"
        << "gpu_per_block_avg_us,gpu_bandwidth_GBps\n";

    for (int blocks = 1; blocks <= max_blocks; blocks *= 2) {
        const size_t bytes = static_cast<size_t>(blocks) * block_bytes;

        for (int stream_count : STREAM_COUNTS) {
            int active_streams = std::min(stream_count, blocks);

            double wall_ms = 0.0;
            double gpu_ms = 0.0;

            for (int i = 0; i < warmup; ++i) {
                copy_blocks_h2d_async(
                    d_dst, h_src, blocks, block_elems, block_bytes,
                    streams, stream_count,
                    start_ev, stop_ev, join_evs,
                    wall_ms, gpu_ms
                );
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> wall_times;
            std::vector<double> gpu_times;
            wall_times.reserve(iterations);
            gpu_times.reserve(iterations);

            for (int i = 0; i < iterations; ++i) {
                copy_blocks_h2d_async(
                    d_dst, h_src, blocks, block_elems, block_bytes,
                    streams, stream_count,
                    start_ev, stop_ev, join_evs,
                    wall_ms, gpu_ms
                );

                wall_times.push_back(wall_ms);
                gpu_times.push_back(gpu_ms);
            }

            double wall_sum = std::accumulate(
                wall_times.begin(), wall_times.end(), 0.0
            );
            double gpu_sum = std::accumulate(
                gpu_times.begin(), gpu_times.end(), 0.0
            );

            double wall_avg = wall_sum / static_cast<double>(wall_times.size());
            double wall_p50 = percentile(wall_times, 50.0);
            double wall_p99 = percentile(wall_times, 99.0);

            double gpu_avg = gpu_sum / static_cast<double>(gpu_times.size());
            double gpu_p50 = percentile(gpu_times, 50.0);
            double gpu_p99 = percentile(gpu_times, 99.0);

            // 호출 오버헤드 추정 = wall-clock 평균 - 순수 GPU 전송 평균.
            // stream/blocks가 많을수록 이 값이 커지면, 병목이 PCIe가 아니라
            // CPU 측 디스패치(cudaMemcpyAsync 호출 횟수)임을 뜻한다.
            double call_overhead_avg = wall_avg - gpu_avg;

            // 대역폭은 순수 GPU 전송 시간 기준으로 계산해야 PCIe 실효 대역폭에
            // 가깝다. wall-clock 기준으로 계산하면 호출 오버헤드 때문에
            // 대역폭이 실제보다 낮게 보인다.
            double gpu_per_block_avg_us =
                (gpu_avg * 1000.0) / static_cast<double>(blocks);

            double gb = static_cast<double>(bytes) / 1e9;
            double gpu_sec = gpu_avg / 1000.0;
            double gpu_bandwidth_gbps = (gpu_sec > 0.0) ? (gb / gpu_sec) : 0.0;

            std::cout
                << "H2D,"
                << "multi_stream_cudaMemcpyAsync,"
                << blocks << ","
                << stream_count << ","
                << active_streams << ","
                << blocks << ","
                << bytes << ","
                << wall_avg << ","
                << wall_p50 << ","
                << wall_p99 << ","
                << gpu_avg << ","
                << gpu_p50 << ","
                << gpu_p99 << ","
                << call_overhead_avg << ","
                << gpu_per_block_avg_us << ","
                << gpu_bandwidth_gbps
                << "\n";
        }
    }

    for (int i = 0; i < MAX_STREAMS; ++i) {
        CUDA_CHECK(cudaEventDestroy(join_evs[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start_ev));
    CUDA_CHECK(cudaEventDestroy(stop_ev));

    for (int i = 0; i < MAX_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_src));

    return 0;
}
