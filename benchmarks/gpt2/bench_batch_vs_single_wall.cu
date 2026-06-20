#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/request.h"
#include "runtime/response.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;
namespace Model = mini_llm::model;

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << what << " failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

double now_ms() {
    using Clock = std::chrono::high_resolution_clock;
    static const auto t0 = Clock::now();
    auto t = Clock::now();
    return std::chrono::duration<double, std::milli>(t - t0).count();
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

void prepare_blocks(
    Rt::Request& req,
    int total_tokens_needed,
    int& next_block_id
) {
    int blocks_per_layer =
        (total_tokens_needed + C::DEFAULT_KV_BLOCK_SIZE - 1) /
        C::DEFAULT_KV_BLOCK_SIZE;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];
        kv.reset();

        for (int b = 0; b < blocks_per_layer; ++b) {
            kv.append_block(next_block_id++);
        }
    }
}

std::vector<std::unique_ptr<Rt::Request>> make_requests(
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    uint64_t request_id_base
) {
    std::vector<std::unique_ptr<Rt::Request>> reqs;
    reqs.reserve(batch_size);

    int next_block_id = 0;
    int total_tokens_needed = prompt_len + max_new_tokens;

    for (int i = 0; i < batch_size; ++i) {
        auto req = std::make_unique<Rt::Request>(
            static_cast<int>(request_id_base + i),
            make_prompt(prompt_len),
            max_new_tokens
        );

        prepare_blocks(*req, total_tokens_needed, next_block_id);
        reqs.push_back(std::move(req));
    }

    if (next_block_id > C::DEFAULT_TOTAL_KV_BLOCKS) {
        std::cerr << "benchmark needs " << next_block_id
                  << " KV blocks, but DEFAULT_TOTAL_KV_BLOCKS is "
                  << C::DEFAULT_TOTAL_KV_BLOCKS << "\n";
        std::exit(1);
    }

    return reqs;
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
    }
}

void apply_decode_responses(
    std::vector<std::unique_ptr<Rt::Request>>& reqs,
    const std::vector<Rt::Response>& responses
) {
    for (size_t i = 0; i < reqs.size(); ++i) {
        int token = 0;
        if (i < responses.size()) {
            token = responses[i].token;
        }

        for (auto& kv : reqs[i]->layer_kv) {
            kv.increment_tokens(1);
        }

        reqs[i]->tokens.push_back(token);
    }
}

struct RunResult {
    double wall_ms = 0.0;
    int output_tokens = 0;
};

RunResult run_batched_once(
    Model::GPT2Model& model,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    uint64_t request_id_base
) {
    auto reqs = make_requests(
        batch_size,
        prompt_len,
        max_new_tokens,
        request_id_base
    );

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before)");

    double start_ms = now_ms();

    std::vector<Rt::Response> prefill_responses = model.prefill(reqs);
    apply_prefill_responses(reqs, prefill_responses);

    for (int step = 1; step < max_new_tokens; ++step) {
        std::vector<Rt::Response> decode_responses = model.decode(reqs);
        apply_decode_responses(reqs, decode_responses);
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after)");

    double end_ms = now_ms();

    RunResult result;
    result.wall_ms = end_ms - start_ms;
    result.output_tokens = batch_size * max_new_tokens;
    return result;
}

RunResult run_sequential_once(
    Model::GPT2Model& model,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    uint64_t request_id_base
) {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before)");

    double start_ms = now_ms();

    int output_tokens = 0;

    for (int i = 0; i < batch_size; ++i) {
        auto reqs = make_requests(
            1,
            prompt_len,
            max_new_tokens,
            request_id_base + static_cast<uint64_t>(i)
        );

        std::vector<Rt::Response> prefill_responses = model.prefill(reqs);
        apply_prefill_responses(reqs, prefill_responses);

        for (int step = 1; step < max_new_tokens; ++step) {
            std::vector<Rt::Response> decode_responses = model.decode(reqs);
            apply_decode_responses(reqs, decode_responses);
        }

        output_tokens += max_new_tokens;
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after)");

    double end_ms = now_ms();

    RunResult result;
    result.wall_ms = end_ms - start_ms;
    result.output_tokens = output_tokens;
    return result;
}

struct AggregateResult {
    double avg_wall_ms = 0.0;
    double tokens_per_sec = 0.0;
};

AggregateResult aggregate_runs(
    const std::string& mode,
    Model::GPT2Model& model,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    int iterations
) {
    double total_ms = 0.0;
    int total_output_tokens = 0;

    for (int it = 0; it < iterations; ++it) {
        uint64_t request_id_base =
            static_cast<uint64_t>(1000000 + it * 10000 + batch_size * 100);

        RunResult r;

        if (mode == "batched") {
            r = run_batched_once(
                model,
                batch_size,
                prompt_len,
                max_new_tokens,
                request_id_base
            );
        } else {
            r = run_sequential_once(
                model,
                batch_size,
                prompt_len,
                max_new_tokens,
                request_id_base
            );
        }

        total_ms += r.wall_ms;
        total_output_tokens += r.output_tokens;
    }

    AggregateResult result;
    result.avg_wall_ms = total_ms / static_cast<double>(iterations);
    result.tokens_per_sec =
        static_cast<double>(total_output_tokens) / (total_ms / 1000.0);

    return result;
}

void print_result(
    const std::string& mode,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    int iterations,
    const AggregateResult& r
) {
    std::cout
        << batch_size << ","
        << prompt_len << ","
        << max_new_tokens << ","
        << iterations << ","
        << mode << ","
        << r.avg_wall_ms << ","
        << r.tokens_per_sec << "\n";
}

} // namespace

int main(int argc, char** argv) {
    int prompt_len = 5;
    int max_new_tokens = 8;
    int iterations = 5;
    int warmup = 1;

    if (argc >= 2) {
        prompt_len = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        max_new_tokens = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        iterations = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        warmup = std::atoi(argv[4]);
    }

    if (prompt_len <= 0 || max_new_tokens <= 0 || iterations <= 0 || warmup < 0) {
        std::cerr << "usage: ./bench_batch_vs_single_wall "
                  << "[prompt_len] [max_new_tokens] [iterations] [warmup]\n";
        return 1;
    }

    auto& model = Model::GPT2Model::get();

    for (int i = 0; i < warmup; ++i) {
        (void)run_batched_once(
            model,
            1,
            prompt_len,
            max_new_tokens,
            9000000 + i
        );
    }

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16};

    std::cout
        << "batch_size,prompt_len,max_new_tokens,iterations,mode,"
        << "avg_total_ms,tokens_per_sec\n";

    for (int batch_size : batch_sizes) {
        AggregateResult seq = aggregate_runs(
            "sequential",
            model,
            batch_size,
            prompt_len,
            max_new_tokens,
            iterations
        );

        AggregateResult batched = aggregate_runs(
            "batched",
            model,
            batch_size,
            prompt_len,
            max_new_tokens,
            iterations
        );

        print_result(
            "sequential",
            batch_size,
            prompt_len,
            max_new_tokens,
            iterations,
            seq
        );

        print_result(
            "batched",
            batch_size,
            prompt_len,
            max_new_tokens,
            iterations,
            batched
        );
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(final)");
    return 0;
}
