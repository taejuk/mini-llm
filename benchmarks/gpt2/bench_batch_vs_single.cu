#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/request.h"
#include "runtime/response.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
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

float measure_cuda_ms(const std::function<void()>& fn) {
    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    fn();
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    return ms;
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

struct RunMetrics {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    std::vector<double> ttft_ms;
};

RunMetrics run_batched_once(
    Model::GPT2Model& model,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    uint64_t request_id_base
) {
    RunMetrics m;
    auto reqs = make_requests(
        batch_size,
        prompt_len,
        max_new_tokens,
        request_id_base
    );

    std::vector<Rt::Response> prefill_responses;
    m.prefill_ms = measure_cuda_ms([&] {
        prefill_responses = model.prefill(reqs);
    });
    apply_prefill_responses(reqs, prefill_responses);

    m.ttft_ms.assign(batch_size, m.prefill_ms);

    for (int step = 1; step < max_new_tokens; ++step) {
        std::vector<Rt::Response> decode_responses;
        float step_ms = measure_cuda_ms([&] {
            decode_responses = model.decode(reqs);
        });

        m.decode_ms += step_ms;
        apply_decode_responses(reqs, decode_responses);
    }

    m.total_ms = m.prefill_ms + m.decode_ms;
    return m;
}

RunMetrics run_sequential_once(
    Model::GPT2Model& model,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    uint64_t request_id_base
) {
    RunMetrics m;
    double elapsed_ms = 0.0;

    for (int i = 0; i < batch_size; ++i) {
        auto reqs = make_requests(
            1,
            prompt_len,
            max_new_tokens,
            request_id_base + static_cast<uint64_t>(i)
        );

        std::vector<Rt::Response> prefill_responses;
        float prefill_ms = measure_cuda_ms([&] {
            prefill_responses = model.prefill(reqs);
        });

        m.prefill_ms += prefill_ms;
        elapsed_ms += prefill_ms;
        m.ttft_ms.push_back(elapsed_ms);
        apply_prefill_responses(reqs, prefill_responses);

        for (int step = 1; step < max_new_tokens; ++step) {
            std::vector<Rt::Response> decode_responses;
            float step_ms = measure_cuda_ms([&] {
                decode_responses = model.decode(reqs);
            });

            m.decode_ms += step_ms;
            elapsed_ms += step_ms;
            apply_decode_responses(reqs, decode_responses);
        }
    }

    m.total_ms = elapsed_ms;
    return m;
}

double mean(const std::vector<double>& xs) {
    if (xs.empty()) {
        return 0.0;
    }

    double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    return sum / static_cast<double>(xs.size());
}

double percentile(std::vector<double> xs, double p) {
    if (xs.empty()) {
        return 0.0;
    }

    std::sort(xs.begin(), xs.end());
    size_t idx = static_cast<size_t>((p / 100.0) * static_cast<double>(xs.size() - 1));
    return xs[idx];
}

struct AggregateMetrics {
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    std::vector<double> ttft_ms;
};

AggregateMetrics aggregate_runs(
    const std::string& mode,
    Model::GPT2Model& model,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    int iterations
) {
    AggregateMetrics agg;

    for (int it = 0; it < iterations; ++it) {
        uint64_t request_id_base =
            static_cast<uint64_t>(1000000 + it * 10000 + batch_size * 100);

        RunMetrics m;
        if (mode == "batched") {
            m = run_batched_once(
                model,
                batch_size,
                prompt_len,
                max_new_tokens,
                request_id_base
            );
        } else {
            m = run_sequential_once(
                model,
                batch_size,
                prompt_len,
                max_new_tokens,
                request_id_base
            );
        }

        agg.prefill_ms += m.prefill_ms;
        agg.decode_ms += m.decode_ms;
        agg.total_ms += m.total_ms;
        agg.ttft_ms.insert(agg.ttft_ms.end(), m.ttft_ms.begin(), m.ttft_ms.end());
    }

    agg.prefill_ms /= static_cast<double>(iterations);
    agg.decode_ms /= static_cast<double>(iterations);
    agg.total_ms /= static_cast<double>(iterations);
    return agg;
}

void print_result(
    const std::string& mode,
    int batch_size,
    int prompt_len,
    int max_new_tokens,
    int iterations,
    const AggregateMetrics& m
) {
    double total_tokens = static_cast<double>(batch_size * max_new_tokens);
    double tokens_per_sec = total_tokens / (m.total_ms / 1000.0);

    std::cout
        << batch_size << ","
        << prompt_len << ","
        << max_new_tokens << ","
        << iterations << ","
        << mode << ","
        << mean(m.ttft_ms) << ","
        << percentile(m.ttft_ms, 50.0) << ","
        << percentile(m.ttft_ms, 99.0) << ","
        << m.prefill_ms << ","
        << m.decode_ms << ","
        << m.total_ms << ","
        << tokens_per_sec << "\n";
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
        std::cerr << "usage: ./bench_batch_vs_single [prompt_len] [max_new_tokens] [iterations] [warmup]\n";
        return 1;
    }

    auto& model = Model::GPT2Model::get();

    for (int i = 0; i < warmup; ++i) {
        (void)run_batched_once(model, 1, prompt_len, max_new_tokens, 9000000 + i);
    }

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16};

    std::cout
        << "batch_size,prompt_len,max_new_tokens,iterations,mode,"
        << "ttft_avg_ms,ttft_p50_ms,ttft_p99_ms,"
        << "prefill_ms,decode_ms,total_ms,tokens_per_sec\n";

    for (int batch_size : batch_sizes) {
        AggregateMetrics seq = aggregate_runs(
            "sequential",
            model,
            batch_size,
            prompt_len,
            max_new_tokens,
            iterations
        );

        AggregateMetrics batched = aggregate_runs(
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
