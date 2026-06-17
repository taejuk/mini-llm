#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/block_manager.h"
#include "runtime/cpu_pool.cuh"
#include "runtime/pool.cuh"
#include "runtime/real_kv_allocator.h"
#include "runtime/request.h"
#include "runtime/response.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;
namespace Model = mini_llm::model;

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

double avg(const std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    }

    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
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

std::vector<int> make_prompt(int prompt_len) {
    std::vector<int> base = {15496, 11, 616, 1438, 318};
    std::vector<int> prompt;
    prompt.reserve(prompt_len);

    for (int i = 0; i < prompt_len; ++i) {
        prompt.push_back(base[i % static_cast<int>(base.size())]);
    }

    return prompt;
}

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

    for (int e = tail_start + threadIdx.x; e < block_elems; e += blockDim.x) {
        dst[e] = src[e];
    }
}

void launch_swap_in_vec4(
    float* d_pool,
    float* d_staging,
    const float* h_contig,
    const int* d_dst_idx,
    int blocks,
    size_t block_elems,
    size_t bytes,
    cudaStream_t stream
) {
    CUDA_CHECK(cudaMemcpyAsync(
        d_staging,
        h_contig,
        bytes,
        cudaMemcpyHostToDevice,
        stream
    ));

    constexpr int threads = 256;
    int grid = blocks;

    scatter_blocks_vec4_kernel<<<grid, threads, 0, stream>>>(
        d_pool,
        d_staging,
        d_dst_idx,
        blocks,
        static_cast<int>(block_elems)
    );

    CUDA_CHECK(cudaGetLastError());
}

void apply_prefill_responses(
    std::vector<std::unique_ptr<Rt::Request>>& reqs,
    const std::vector<Rt::Response>& responses
) {
    for (size_t i = 0; i < reqs.size(); ++i) {
        int token = 0;

        if (i < responses.size()) {
            token = responses[i].token;
        }

        for (auto& kv : reqs[i]->layer_kv) {
            kv.set_num_tokens(reqs[i]->prompts_len);
        }

        reqs[i]->tokens.push_back(token);
        reqs[i]->state = Rt::RequestState::DecodeReady;
        reqs[i]->kv_residency = Rt::KvCacheResidency::Gpu;
    }
}

std::vector<std::unique_ptr<Rt::Request>> make_decode_requests(
    int batch_size,
    int prompt_len,
    Rt::RealKvAllocator& allocator,
    Model::GPT2Model& model
) {
    std::vector<std::unique_ptr<Rt::Request>> reqs;
    reqs.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        auto req = std::make_unique<Rt::Request>(
            1000 + i,
            make_prompt(prompt_len),
            16
        );

        bool ok = allocator.allocate_prefill(*req);

        if (!ok) {
            std::cerr << "allocate_prefill failed for decode request\n";
            std::exit(1);
        }

        reqs.push_back(std::move(req));
    }

    std::vector<Rt::Response> responses = model.prefill(reqs);
    CUDA_CHECK(cudaDeviceSynchronize());

    apply_prefill_responses(reqs, responses);

    // Important:
    // decode writes the new token KV at cached_tokens position.
    // If prompt_len is exactly on a KV block boundary, decode needs one more
    // KV block per layer. This mirrors scheduler behavior.
    for (auto& req : reqs) {
        bool ok = allocator.allocate_decode(*req);

        if (!ok) {
            std::cerr << "allocate_decode failed for decode request\n";
            std::exit(1);
        }
    }

    return reqs;
}

std::vector<int> collect_used_blocks(
    const std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    std::vector<int> used;

    for (const auto& req : reqs) {
        for (const auto& kv : req->layer_kv) {
            for (int block_id : kv.block_table_) {
                used.push_back(block_id);
            }
        }
    }

    return used;
}

std::vector<int> make_swap_destination_blocks(
    int blocks,
    int total_pool_blocks,
    const std::vector<int>& used_blocks,
    std::mt19937& rng
) {
    std::vector<char> used(total_pool_blocks, 0);

    for (int id : used_blocks) {
        if (id >= 0 && id < total_pool_blocks) {
            used[id] = 1;
        }
    }

    std::vector<int> candidates;
    candidates.reserve(total_pool_blocks);

    for (int id = 0; id < total_pool_blocks; ++id) {
        if (!used[id]) {
            candidates.push_back(id);
        }
    }

    if (static_cast<int>(candidates.size()) < blocks) {
        std::cerr
            << "not enough free destination blocks for swap benchmark. "
            << "needed=" << blocks
            << ", available=" << candidates.size()
            << "\n";
        std::exit(1);
    }

    std::shuffle(candidates.begin(), candidates.end(), rng);
    candidates.resize(blocks);

    return candidates;
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 100;
    int warmup = 10;
    int batch_size = 1;
    int prompt_len = 5;
    unsigned int seed = 1234;

    if (argc >= 2) {
        iterations = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        warmup = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        batch_size = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        prompt_len = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        seed = static_cast<unsigned int>(std::atoi(argv[5]));
    }

    if (iterations <= 0 || warmup < 0) {
        std::cerr
            << "usage: ./bench_real_decode_swap_overlap "
            << "[iterations] [warmup] [decode_batch_size] "
            << "[decode_prompt_len] [seed]\n";
        return 1;
    }

    if (batch_size <= 0 || batch_size > C::MAX_BATCH_NUM) {
        std::cerr << "batch_size must be in [1, "
                  << C::MAX_BATCH_NUM << "]\n";
        return 1;
    }

    if (prompt_len <= 0 || prompt_len >= C::MAX_SEQ) {
        std::cerr << "prompt_len must be in [1, "
                  << C::MAX_SEQ - 1 << "]\n";
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
    std::cout << "asyncEngineCount=" << prop.asyncEngineCount << "\n";
    std::cout << "concurrentKernels=" << prop.concurrentKernels << "\n";
    std::cout << "block_elems=" << block_elems << "\n";
    std::cout << "block_bytes=" << block_bytes << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "decode_batch_size=" << batch_size << "\n";
    std::cout << "decode_prompt_len=" << prompt_len << "\n";
    std::cout << "NOTE: decode uses current GPT2Model::decode() as-is.\n";
    std::cout << "NOTE: decode requests allocate decode KV blocks before timing.\n";
    std::cout << "NOTE: swap_in uses H2D to GPU staging + vec4 scatter "
                 "on a non-blocking stream.\n\n";

    Rt::Pool& pool =
        Rt::Pool::getInstance(C::DEFAULT_TOTAL_KV_BLOCKS);

    Rt::CpuPool& cpu_pool =
        Rt::CpuPool::getInstance(C::DEFAULT_TOTAL_CPU_BLOCKS);

    Rt::BlockManager& block_manager =
        Rt::BlockManager::getInstance(pool, cpu_pool);

    (void)block_manager;

    Rt::RealKvAllocator allocator;
    Model::GPT2Model& model = Model::GPT2Model::get();

    auto decode_reqs = make_decode_requests(
        batch_size,
        prompt_len,
        allocator,
        model
    );

    std::vector<int> used_blocks = collect_used_blocks(decode_reqs);

    float* h_contig = nullptr;

    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&h_contig),
        static_cast<size_t>(max_blocks) * block_bytes
    ));

    for (size_t i = 0;
         i < static_cast<size_t>(max_blocks) * block_elems;
         ++i) {
        h_contig[i] = static_cast<float>(i % 1024);
    }

    float* d_staging = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_staging),
        static_cast<size_t>(max_blocks) * block_bytes
    ));

    int* d_dst_idx = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst_idx),
        static_cast<size_t>(max_blocks) * sizeof(int)
    ));

    cudaStream_t swap_stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &swap_stream,
        cudaStreamNonBlocking
    ));

    std::mt19937 rng(seed);

    std::cout
        << "blocks,bytes,"
        << "swap_only_avg_ms,swap_only_p99_ms,"
        << "decode_only_avg_ms,decode_only_p99_ms,"
        << "sequential_avg_ms,sequential_p99_ms,"
        << "overlap_avg_ms,overlap_p99_ms,"
        << "saved_avg_ms,speedup,overlap_efficiency\n";

    for (int blocks : BLOCK_COUNTS) {
        size_t bytes =
            static_cast<size_t>(blocks) * block_bytes;

        std::vector<int> dst_idx = make_swap_destination_blocks(
            blocks,
            pool.total_blocks(),
            used_blocks,
            rng
        );

        CUDA_CHECK(cudaMemcpy(
            d_dst_idx,
            dst_idx.data(),
            static_cast<size_t>(blocks) * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        auto run_swap_only = [&]() -> double {
            CUDA_CHECK(cudaDeviceSynchronize());

            double t0 = now_ms();

            launch_swap_in_vec4(
                pool.pool_start(),
                d_staging,
                h_contig,
                d_dst_idx,
                blocks,
                block_elems,
                bytes,
                swap_stream
            );

            CUDA_CHECK(cudaStreamSynchronize(swap_stream));

            double t1 = now_ms();

            return t1 - t0;
        };

        auto run_decode_only = [&]() -> double {
            CUDA_CHECK(cudaDeviceSynchronize());

            double t0 = now_ms();

            (void)model.decode(decode_reqs);

            CUDA_CHECK(cudaDeviceSynchronize());

            double t1 = now_ms();

            return t1 - t0;
        };

        auto run_sequential = [&]() -> double {
            CUDA_CHECK(cudaDeviceSynchronize());

            double t0 = now_ms();

            launch_swap_in_vec4(
                pool.pool_start(),
                d_staging,
                h_contig,
                d_dst_idx,
                blocks,
                block_elems,
                bytes,
                swap_stream
            );

            CUDA_CHECK(cudaStreamSynchronize(swap_stream));

            (void)model.decode(decode_reqs);

            CUDA_CHECK(cudaDeviceSynchronize());

            double t1 = now_ms();

            return t1 - t0;
        };

        auto run_overlap = [&]() -> double {
            CUDA_CHECK(cudaDeviceSynchronize());

            double t0 = now_ms();

            launch_swap_in_vec4(
                pool.pool_start(),
                d_staging,
                h_contig,
                d_dst_idx,
                blocks,
                block_elems,
                bytes,
                swap_stream
            );

            // Current real decode implementation is used as-is.
            // It runs on the default stream while swap_in runs on a non-blocking stream.
            (void)model.decode(decode_reqs);

            CUDA_CHECK(cudaStreamSynchronize(swap_stream));
            CUDA_CHECK(cudaDeviceSynchronize());

            double t1 = now_ms();

            return t1 - t0;
        };

        for (int i = 0; i < warmup; ++i) {
            (void)run_swap_only();
            (void)run_decode_only();
            (void)run_sequential();
            (void)run_overlap();
        }

        std::vector<double> swap_only;
        std::vector<double> decode_only;
        std::vector<double> sequential;
        std::vector<double> overlap;

        swap_only.reserve(iterations);
        decode_only.reserve(iterations);
        sequential.reserve(iterations);
        overlap.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            swap_only.push_back(run_swap_only());
            decode_only.push_back(run_decode_only());
            sequential.push_back(run_sequential());
            overlap.push_back(run_overlap());
        }

        Result swap_res = summarize(swap_only);
        Result decode_res = summarize(decode_only);
        Result seq_res = summarize(sequential);
        Result ovl_res = summarize(overlap);

        double saved_avg =
            seq_res.avg_ms - ovl_res.avg_ms;

        double speedup =
            ovl_res.avg_ms > 0.0
                ? seq_res.avg_ms / ovl_res.avg_ms
                : 0.0;

        double max_possible_saved =
            std::min(swap_res.avg_ms, decode_res.avg_ms);

        double overlap_efficiency =
            max_possible_saved > 0.0
                ? saved_avg / max_possible_saved
                : 0.0;

        std::cout
            << blocks << ","
            << bytes << ","
            << swap_res.avg_ms << ","
            << swap_res.p99_ms << ","
            << decode_res.avg_ms << ","
            << decode_res.p99_ms << ","
            << seq_res.avg_ms << ","
            << seq_res.p99_ms << ","
            << ovl_res.avg_ms << ","
            << ovl_res.p99_ms << ","
            << saved_avg << ","
            << speedup << ","
            << overlap_efficiency
            << "\n";
    }

    CUDA_CHECK(cudaStreamDestroy(swap_stream));

    CUDA_CHECK(cudaFree(d_dst_idx));
    CUDA_CHECK(cudaFree(d_staging));
    CUDA_CHECK(cudaFreeHost(h_contig));

    for (auto& req : decode_reqs) {
        allocator.free_request(*req);
    }

    return 0;
}
