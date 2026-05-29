#include "model/gpt2_wmma.cuh"
#include "model/gpt2_common.cuh"
#include "scheduler/scheduler.h"
#include "scheduler/request.h"
#include "vllm/allocator.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <numeric>
#include <string>
#include <vector>

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./weights"
#endif

#define CUDA_CHECK_LOCAL(call)                                             \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            std::exit(1);                                                   \
        }                                                                  \
    } while (0)

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    using ms = std::chrono::duration<double, std::milli>;

    return std::chrono::duration_cast<ms>(
        clock::now().time_since_epoch()
    ).count();
}

static std::vector<int> make_prompt(
    int prompt_len,
    int request_id
) {
    std::vector<int> ids(prompt_len);

    for (int i = 0; i < prompt_len; i++) {
        /*
         * GPT-2 vocab 범위 안의 synthetic token.
         * request마다 token pattern을 조금 다르게 해서 완전히 같은 prompt만
         * 반복하지 않도록 한다.
         */
        ids[i] = 100 + ((request_id * 17 + i) % 1000);
    }

    return ids;
}

static int* copy_prompt_to_device(
    const std::vector<int>& ids
) {
    int* d_ids = nullptr;

    CUDA_CHECK_LOCAL(cudaMalloc(
        &d_ids,
        ids.size() * sizeof(int)
    ));

    CUDA_CHECK_LOCAL(cudaMemcpy(
        d_ids,
        ids.data(),
        ids.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    return d_ids;
}

static double percentile(
    std::vector<double> xs,
    double p
) {
    if (xs.empty()) {
        return 0.0;
    }

    std::sort(xs.begin(), xs.end());

    double rank = (p / 100.0) * (double)(xs.size() - 1);
    size_t lo = (size_t)std::floor(rank);
    size_t hi = (size_t)std::ceil(rank);

    if (lo == hi) {
        return xs[lo];
    }

    double w = rank - (double)lo;
    return xs[lo] * (1.0 - w) + xs[hi] * w;
}

static double mean(
    const std::vector<double>& xs
) {
    if (xs.empty()) {
        return 0.0;
    }

    double s = std::accumulate(
        xs.begin(),
        xs.end(),
        0.0
    );

    return s / (double)xs.size();
}

struct RequestMetric {
    int request_id = -1;
    int prompt_len = 0;
    int output_len = 0;

    double arrival_ms = 0.0;
    double admit_ms = -1.0;
    double first_token_ms = -1.0;
    double finish_ms = -1.0;

    bool completed = false;

    std::future<std::vector<int>> future;
};

struct StepMetric {
    int step = 0;

    int waiting_estimate = 0;
    int running_prefill = 0;
    int running_decode = 0;

    int prefill_batch_size = 0;
    int decode_batch_size = 0;

    int used_blocks_before = 0;
    int free_blocks_before = 0;
    int used_blocks_after = 0;
    int free_blocks_after = 0;

    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double step_ms = 0.0;
};

static void print_usage(
    const char* prog
) {
    fprintf(stderr,
        "Usage: %s [num_requests] [prompt_len] [output_len] [block_size] [total_blocks] [print_step_csv]\n"
        "\n"
        "Example:\n"
        "  %s 32 128 32 16 4096 0\n"
        "\n"
        "Arguments:\n"
        "  num_requests   : number of requests submitted at time 0, default 32\n"
        "  prompt_len     : prompt token length per request, default 128\n"
        "  output_len     : generated tokens per request, default 32\n"
        "  block_size     : paged KV block size, default 16\n"
        "  total_blocks   : total physical KV blocks, default 4096\n"
        "  print_step_csv : 1 to print per-step scheduler stats, default 0\n",
        prog,
        prog);
}

int main(
    int argc,
    char** argv
) {
    int num_requests = 32;
    int prompt_len   = 128;
    int output_len   = 32;
    int block_size   = 16;
    int total_blocks = 4096;
    int print_step_csv = 0;

    if (argc >= 2) {
        num_requests = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        prompt_len = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        output_len = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        block_size = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        total_blocks = std::atoi(argv[5]);
    }

    if (argc >= 7) {
        print_step_csv = std::atoi(argv[6]);
    }

    if (argc > 7) {
        print_usage(argv[0]);
        return 1;
    }

    if (num_requests <= 0) {
        fprintf(stderr, "Invalid num_requests=%d\n", num_requests);
        return 1;
    }

    if (prompt_len <= 0 || prompt_len > MAX_SEQ) {
        fprintf(stderr,
                "Invalid prompt_len=%d. It must be in [1, %d].\n",
                prompt_len,
                MAX_SEQ);
        return 1;
    }

    if (output_len <= 0) {
        fprintf(stderr, "Invalid output_len=%d\n", output_len);
        return 1;
    }

    if (prompt_len + output_len > MAX_SEQ) {
        fprintf(stderr,
                "prompt_len + output_len exceeds MAX_SEQ: %d + %d > %d\n",
                prompt_len,
                output_len,
                MAX_SEQ);
        return 1;
    }

    if (block_size <= 0 || total_blocks <= 0) {
        fprintf(stderr,
                "Invalid block_size=%d or total_blocks=%d\n",
                block_size,
                total_blocks);
        return 1;
    }

    int blocks_per_request =
        ((prompt_len + output_len + block_size - 1) / block_size) * N_LAYERS;

    int theoretical_total_blocks_needed =
        blocks_per_request * num_requests;

    fprintf(stderr,
            "[config] num_requests=%d prompt_len=%d output_len=%d block_size=%d total_blocks=%d\n",
            num_requests,
            prompt_len,
            output_len,
            block_size,
            total_blocks);

    fprintf(stderr,
            "[config] blocks_per_request≈%d, total_if_all_running≈%d\n",
            blocks_per_request,
            theoretical_total_blocks_needed);

    if (total_blocks < blocks_per_request) {
        fprintf(stderr,
                "[error] total_blocks=%d is smaller than blocks needed by one request=%d\n",
                total_blocks,
                blocks_per_request);
        return 1;
    }

    if (total_blocks < theoretical_total_blocks_needed) {
        fprintf(stderr,
                "[warning] total_blocks is not enough to keep all requests running simultaneously. "
                "Some requests may wait in scheduler.\n");
    }

    /*
     * BlockAllocator singleton 초기화.
     *
     * 주의:
     * 현재 BlockAllocator는 singleton이고 첫 getInstance()에서만 크기가 정해진다.
     * 같은 process 안에서 total_blocks를 바꾸며 반복 실험하면 안 된다.
     * total_blocks sweep은 실행 파일을 여러 번 실행하는 방식으로 해야 한다.
     */
    BlockAllocator::getInstance(
        total_blocks,
        block_size,
        D_MODEL
    );

    GPT2ModelWMMA::init(
        WEIGHTS_DIR,
        block_size
    );

    GPT2ModelWMMA& model = GPT2ModelWMMA::get();

    Scheduler scheduler(
        block_size,
        D_MODEL
    );

    std::vector<RequestMetric> metrics;
    metrics.resize(num_requests);

    double bench_start_ms = now_ms();

    /*
     * Offline scenario:
     * 모든 request가 arrival_ms=0에 동시에 들어온 것으로 본다.
     */
    for (int i = 0; i < num_requests; i++) {
        std::vector<int> prompt =
            make_prompt(
                prompt_len,
                i
            );

        metrics[i].request_id = i;
        metrics[i].prompt_len = prompt_len;
        metrics[i].output_len = output_len;
        metrics[i].arrival_ms = 0.0;

        metrics[i].future =
            scheduler.submit(
                std::move(prompt),
                output_len
            );
    }

    int completed = 0;
    int step = 0;

    std::vector<StepMetric> step_metrics;
    step_metrics.reserve((size_t)output_len * num_requests + 16);

    if (print_step_csv) {
        printf("step,prefill_batch,decode_batch,free_blocks_before,used_blocks_before,free_blocks_after,used_blocks_after,prefill_ms,decode_ms,step_ms\n");
    }

    while (!scheduler.all_done()) {
        double step_start_ms = now_ms();

        StepMetric sm;
        sm.step = step;

        int free_before =
            BlockAllocator::getInstance().get_num_free_blocks();

        sm.free_blocks_before = free_before;
        sm.used_blocks_before = total_blocks - free_before;

        ScheduleBatch batch =
            scheduler.schedule();

        sm.prefill_batch_size = (int)batch.prefill_reqs.size();
        sm.decode_batch_size  = (int)batch.decode_reqs.size();

        /*
         * deadlock 방지:
         * schedule 결과가 비었는데 all_done이 아니면, 보통 block 부족 때문에
         * waiting request가 admit되지 못한 상황이다.
         */
        if (batch.prefill_reqs.empty() &&
            batch.decode_reqs.empty()) {
            fprintf(stderr,
                    "[error] scheduler returned empty batch while not all_done. "
                    "This likely means KV blocks are insufficient.\n");
            fprintf(stderr,
                    "[debug] free_blocks=%d total_blocks=%d blocks_per_request≈%d\n",
                    free_before,
                    total_blocks,
                    blocks_per_request);
            return 1;
        }

        std::vector<int> next_tokens;
        next_tokens.reserve(
            batch.prefill_reqs.size() +
            batch.decode_reqs.size()
        );

        /*
         * 1. Prefill phase
         *
         * 현재 mini-llm은 prefill batch가 아직 없으므로,
         * prefill request들을 순차 실행한다.
         */
        double prefill_start_ms = now_ms();

        for (Request* req : batch.prefill_reqs) {
            int req_id = req->id;

            if (req_id >= 0 && req_id < num_requests) {
                if (metrics[req_id].admit_ms < 0.0) {
                    metrics[req_id].admit_ms =
                        now_ms() - bench_start_ms;
                }
            }

            int* d_ids =
                copy_prompt_to_device(
                    req->prompt_ids
                );

            int next_tok =
                model.prefill(
                    d_ids,
                    (int)req->prompt_ids.size(),
                    req->layer_kv
                );

            CUDA_CHECK_LOCAL(cudaFree(d_ids));

            next_tokens.push_back(next_tok);

            if (req_id >= 0 && req_id < num_requests) {
                /*
                 * prefill 결과가 첫 generated token이라고 본다.
                 */
                if (metrics[req_id].first_token_ms < 0.0) {
                    metrics[req_id].first_token_ms =
                        now_ms() - bench_start_ms;
                }
            }
        }

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        double prefill_end_ms = now_ms();
        sm.prefill_ms = prefill_end_ms - prefill_start_ms;

        /*
         * 2. Decode phase
         *
         * No-batch decode version.
         *
         * 기존 batch benchmark는:
         *   model.batch_decode(batch.decode_reqs)
         *
         * 를 사용해서 여러 request의 decode 1 step을 한 번에 처리한다.
         *
         * 이 파일에서는 비교 실험을 위해 decode request들을 하나씩 순차 실행한다.
         */
        double decode_start_ms = now_ms();

        for (Request* req : batch.decode_reqs) {
            int input_token =
                req->output_ids.empty()
                    ? req->prompt_ids.back()
                    : req->output_ids.back();

            int next_tok =
                model.decode_step(
                    input_token,
                    req->layer_kv
                );

            next_tokens.push_back(next_tok);
        }

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        double decode_end_ms = now_ms();
        sm.decode_ms = decode_end_ms - decode_start_ms;

        /*
         * 3. Scheduler update
         *
         * next_tokens 순서:
         *   prefill_reqs 먼저,
         *   decode_reqs 다음.
         */
        scheduler.update(
            batch,
            next_tokens
        );

        /*
         * 4. 완료된 future 확인
         */
        for (int i = 0; i < num_requests; i++) {
            if (metrics[i].completed) {
                continue;
            }

            auto status =
                metrics[i].future.wait_for(
                    std::chrono::milliseconds(0)
                );

            if (status == std::future_status::ready) {
                std::vector<int> result =
                    metrics[i].future.get();

                metrics[i].finish_ms =
                    now_ms() - bench_start_ms;

                metrics[i].completed = true;
                completed++;

                if ((int)result.size() != output_len) {
                    fprintf(stderr,
                            "[warning] request %d generated %zu tokens, expected %d\n",
                            i,
                            result.size(),
                            output_len);
                }
            }
        }

        int free_after =
            BlockAllocator::getInstance().get_num_free_blocks();

        sm.free_blocks_after = free_after;
        sm.used_blocks_after = total_blocks - free_after;
        sm.step_ms = now_ms() - step_start_ms;

        step_metrics.push_back(sm);

        if (print_step_csv) {
            printf("%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
                   sm.step,
                   sm.prefill_batch_size,
                   sm.decode_batch_size,
                   sm.free_blocks_before,
                   sm.used_blocks_before,
                   sm.free_blocks_after,
                   sm.used_blocks_after,
                   sm.prefill_ms,
                   sm.decode_ms,
                   sm.step_ms);
        }

        step++;

        if (step > 1000000) {
            fprintf(stderr,
                    "[error] too many scheduler steps. aborting.\n");
            return 1;
        }
    }

    CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

    double bench_end_ms = now_ms();
    double total_time_ms = bench_end_ms - bench_start_ms;

    std::vector<double> ttft_ms;
    std::vector<double> e2e_ms;
    std::vector<double> tpot_ms;

    ttft_ms.reserve(num_requests);
    e2e_ms.reserve(num_requests);
    tpot_ms.reserve(num_requests);

    for (const auto& m : metrics) {
        if (!m.completed) {
            continue;
        }

        double ttft =
            m.first_token_ms - m.arrival_ms;

        double e2e =
            m.finish_ms - m.arrival_ms;

        double tpot = 0.0;

        if (m.output_len > 1) {
            tpot =
                (m.finish_ms - m.first_token_ms) /
                (double)(m.output_len - 1);
        }

        ttft_ms.push_back(ttft);
        e2e_ms.push_back(e2e);
        tpot_ms.push_back(tpot);
    }

    int total_prompt_tokens =
        num_requests * prompt_len;

    int total_output_tokens =
        completed * output_len;

    int total_tokens =
        total_prompt_tokens + total_output_tokens;

    double seconds =
        total_time_ms / 1000.0;

    double request_throughput =
        seconds > 0.0 ? (double)completed / seconds : 0.0;

    double output_token_throughput =
        seconds > 0.0 ? (double)total_output_tokens / seconds : 0.0;

    double total_token_throughput =
        seconds > 0.0 ? (double)total_tokens / seconds : 0.0;

    int peak_used_blocks = 0;

    for (const auto& sm : step_metrics) {
        peak_used_blocks =
            std::max(
                peak_used_blocks,
                sm.used_blocks_after
            );
    }

    int final_free_blocks =
        BlockAllocator::getInstance().get_num_free_blocks();

    fprintf(stderr, "\n===== Offline GPT-2 WMMA Serving Benchmark No Decode Batch =====\n");
    fprintf(stderr, "num_requests,%d\n", num_requests);
    fprintf(stderr, "completed_requests,%d\n", completed);
    fprintf(stderr, "prompt_len,%d\n", prompt_len);
    fprintf(stderr, "output_len,%d\n", output_len);
    fprintf(stderr, "block_size,%d\n", block_size);
    fprintf(stderr, "total_blocks,%d\n", total_blocks);
    fprintf(stderr, "blocks_per_request_estimate,%d\n", blocks_per_request);
    fprintf(stderr, "peak_used_blocks,%d\n", peak_used_blocks);
    fprintf(stderr, "final_free_blocks,%d\n", final_free_blocks);
    fprintf(stderr, "scheduler_steps,%d\n", step);
    fprintf(stderr, "total_time_ms,%.6f\n", total_time_ms);
    fprintf(stderr, "request_throughput_req_per_s,%.6f\n", request_throughput);
    fprintf(stderr, "output_token_throughput_tok_per_s,%.6f\n", output_token_throughput);
    fprintf(stderr, "total_token_throughput_tok_per_s,%.6f\n", total_token_throughput);

    fprintf(stderr, "ttft_mean_ms,%.6f\n", mean(ttft_ms));
    fprintf(stderr, "ttft_p50_ms,%.6f\n", percentile(ttft_ms, 50.0));
    fprintf(stderr, "ttft_p90_ms,%.6f\n", percentile(ttft_ms, 90.0));
    fprintf(stderr, "ttft_p95_ms,%.6f\n", percentile(ttft_ms, 95.0));
    fprintf(stderr, "ttft_p99_ms,%.6f\n", percentile(ttft_ms, 99.0));

    fprintf(stderr, "tpot_mean_ms,%.6f\n", mean(tpot_ms));
    fprintf(stderr, "tpot_p50_ms,%.6f\n", percentile(tpot_ms, 50.0));
    fprintf(stderr, "tpot_p90_ms,%.6f\n", percentile(tpot_ms, 90.0));
    fprintf(stderr, "tpot_p95_ms,%.6f\n", percentile(tpot_ms, 95.0));
    fprintf(stderr, "tpot_p99_ms,%.6f\n", percentile(tpot_ms, 99.0));

    fprintf(stderr, "e2e_mean_ms,%.6f\n", mean(e2e_ms));
    fprintf(stderr, "e2e_p50_ms,%.6f\n", percentile(e2e_ms, 50.0));
    fprintf(stderr, "e2e_p90_ms,%.6f\n", percentile(e2e_ms, 90.0));
    fprintf(stderr, "e2e_p95_ms,%.6f\n", percentile(e2e_ms, 95.0));
    fprintf(stderr, "e2e_p99_ms,%.6f\n", percentile(e2e_ms, 99.0));

    /*
     * machine-readable summary.
     * stderr에는 사람이 읽는 summary,
     * stdout에는 CSV 한 줄을 출력한다.
     *
     * print_step_csv=1이면 stdout에 step CSV가 먼저 찍히므로,
     * summary만 파싱하고 싶으면 print_step_csv=0으로 실행하자.
     */
    if (!print_step_csv) {
        printf("num_requests,prompt_len,output_len,block_size,total_blocks,completed,total_time_ms,req_per_s,out_tok_per_s,total_tok_per_s,peak_used_blocks,ttft_mean,ttft_p50,ttft_p95,tpot_mean,tpot_p50,tpot_p95,e2e_mean,e2e_p50,e2e_p95\n");

        printf("%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               num_requests,
               prompt_len,
               output_len,
               block_size,
               total_blocks,
               completed,
               total_time_ms,
               request_throughput,
               output_token_throughput,
               total_token_throughput,
               peak_used_blocks,
               mean(ttft_ms),
               percentile(ttft_ms, 50.0),
               percentile(ttft_ms, 95.0),
               mean(tpot_ms),
               percentile(tpot_ms, 50.0),
               percentile(tpot_ms, 95.0),
               mean(e2e_ms),
               percentile(e2e_ms, 50.0),
               percentile(e2e_ms, 95.0));
    }

    return 0;
}
