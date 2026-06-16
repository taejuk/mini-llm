#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/request.h"
#include "runtime/response.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
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

std::vector<int> make_prompt(int token_len) {
    std::vector<int> base = {15496, 11, 616, 1438, 318};
    std::vector<int> prompt;
    prompt.reserve(token_len);

    for (int i = 0; i < token_len; ++i) {
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

std::vector<std::unique_ptr<Rt::Request>> make_one_request(
    int token_len,
    int max_new_tokens,
    uint64_t request_id
) {
    std::vector<std::unique_ptr<Rt::Request>> reqs;
    reqs.reserve(1);

    auto req = std::make_unique<Rt::Request>(
        static_cast<int>(request_id),
        make_prompt(token_len),
        max_new_tokens
    );

    int next_block_id = 0;
    int total_tokens_needed = token_len + max_new_tokens;

    prepare_blocks(*req, total_tokens_needed, next_block_id);

    if (next_block_id > C::DEFAULT_TOTAL_KV_BLOCKS) {
        std::cerr << "benchmark needs " << next_block_id
                  << " KV blocks, but DEFAULT_TOTAL_KV_BLOCKS is "
                  << C::DEFAULT_TOTAL_KV_BLOCKS << "\n";
        std::exit(1);
    }

    reqs.push_back(std::move(req));
    return reqs;
}

void apply_prefill_response(
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

double benchmark_prefill_avg_ms(
    Model::GPT2Model& model,
    int token_len,
    int iterations
) {
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto reqs = make_one_request(
            token_len,
            2,
            static_cast<uint64_t>(1000000 + token_len * 10000 + i)
        );

        float ms = measure_cuda_ms([&] {
            (void)model.prefill(reqs);
        });

        total_ms += static_cast<double>(ms);
    }

    return total_ms / static_cast<double>(iterations);
}

double benchmark_decode_avg_ms(
    Model::GPT2Model& model,
    int token_len,
    int iterations
) {
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto reqs = make_one_request(
            token_len,
            2,
            static_cast<uint64_t>(2000000 + token_len * 10000 + i)
        );

        std::vector<Rt::Response> prefill_responses = model.prefill(reqs);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(prefill)");
        apply_prefill_response(reqs, prefill_responses);

        float ms = measure_cuda_ms([&] {
            (void)model.decode(reqs);
        });

        total_ms += static_cast<double>(ms);
    }

    return total_ms / static_cast<double>(iterations);
}

void warmup(
    Model::GPT2Model& model,
    int token_len,
    int warmup_iters
) {
    for (int i = 0; i < warmup_iters; ++i) {
        auto reqs = make_one_request(
            token_len,
            2,
            static_cast<uint64_t>(3000000 + token_len * 10000 + i)
        );

        auto prefill_responses = model.prefill(reqs);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup_prefill)");

        apply_prefill_response(reqs, prefill_responses);

        (void)model.decode(reqs);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup_decode)");
    }
}

} // namespace

int main(int argc, char** argv) {
    int max_token_len = 1024;
    int iterations = 10;
    int warmup_iters = 2;

    if (argc >= 2) {
        max_token_len = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        iterations = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        warmup_iters = std::atoi(argv[3]);
    }

    if (max_token_len <= 0 || iterations <= 0 || warmup_iters < 0) {
        std::cerr
            << "usage: ./bench_prefill_decode_by_len "
            << "[max_token_len] [iterations] [warmup_iters]\n";
        return 1;
    }

    auto& model = Model::GPT2Model::get();

    std::cout << "token_len,prefill_avg_ms,decode_avg_ms\n";

    for (int token_len = 1; token_len <= max_token_len; token_len *= 2) {
        warmup(model, token_len, warmup_iters);

        double prefill_avg_ms = benchmark_prefill_avg_ms(
            model,
            token_len,
            iterations
        );

        double decode_avg_ms = benchmark_decode_avg_ms(
            model,
            token_len,
            iterations
        );

        std::cout
            << token_len << ","
            << prefill_avg_ms << ","
            << decode_avg_ms
            << "\n";
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(final)");
    return 0;
}