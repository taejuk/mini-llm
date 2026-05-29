#include "model/gpt2_wmma.cuh"
#include "model/gpt2_common.cuh"
#include "vllm/allocator.cuh"
#include "vllm/kv_cache.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static float elapsed_gpu_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK_LOCAL(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static std::vector<int> make_prompt(int prompt_len) {
    std::vector<int> ids(prompt_len);

    for (int i = 0; i < prompt_len; i++) {
        /*
         * GPT-2 vocab 범위 안의 deterministic dummy token.
         * correctness 비교가 목적이면 tokenizer로 얻은 실제 token id 배열로 바꾸면 됨.
         */
        ids[i] = 100 + (i % 1000);
    }

    return ids;
}

static int* copy_prompt_to_device(const std::vector<int>& ids) {
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

static std::vector<PagedKVCache> make_layer_kv(
    int block_size,
    int hidden_dim,
    int request_id
) {
    std::vector<PagedKVCache> layer_kv;
    layer_kv.reserve(N_LAYERS);

    for (int l = 0; l < N_LAYERS; l++) {
        layer_kv.emplace_back(
            block_size,
            hidden_dim,
            request_id
        );
    }

    return layer_kv;
}

static void free_layer_kv(std::vector<PagedKVCache>& layer_kv) {
    for (auto& kv : layer_kv) {
        kv.free_all();
    }
}

static void check_layer_kv_consistency(
    const std::vector<PagedKVCache>& layer_kv,
    int expected_len,
    const char* where
) {
    if ((int)layer_kv.size() != N_LAYERS) {
        fprintf(stderr,
                "%s: layer_kv.size()=%zu, expected %d\n",
                where,
                layer_kv.size(),
                N_LAYERS);
        std::exit(1);
    }

    for (int l = 0; l < N_LAYERS; l++) {
        int len = layer_kv[l].get_num_tokens();

        if (len != expected_len) {
            fprintf(stderr,
                    "%s: layer %d KV length=%d, expected=%d\n",
                    where,
                    l,
                    len,
                    expected_len);
            std::exit(1);
        }
    }
}

static int count_layer_kv_blocks(
    const std::vector<PagedKVCache>& layer_kv
) {
    int total = 0;

    for (const auto& kv : layer_kv) {
        total += kv.get_num_blocks();
    }

    return total;
}

static double mean(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;

    double s = std::accumulate(xs.begin(), xs.end(), 0.0);
    return s / (double)xs.size();
}

static double percentile(std::vector<double> xs, double q) {
    if (xs.empty()) return 0.0;

    std::sort(xs.begin(), xs.end());

    double pos = q * (double)(xs.size() - 1);
    int lo = (int)std::floor(pos);
    int hi = (int)std::ceil(pos);

    if (lo == hi) {
        return xs[lo];
    }

    double w = pos - (double)lo;
    return xs[lo] * (1.0 - w) + xs[hi] * w;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s [warmup_runs] [measure_runs] [block_size] [total_blocks] [prompt_len...]\n"
        "\n"
        "Example:\n"
        "  %s 3 10 16 8192 32 64 128 256 512 1024\n"
        "\n"
        "Arguments:\n"
        "  warmup_runs   : warmup prefill runs per prompt_len, default 3\n"
        "  measure_runs  : measured prefill runs per prompt_len, default 10\n"
        "  block_size    : paged KV block size, default 16\n"
        "  total_blocks  : total physical KV blocks, default 8192\n"
        "  prompt_len... : prompt lengths to sweep\n",
        prog,
        prog);
}

static void run_single_prefill_once(
    GPT2ModelWMMA& model,
    const std::vector<int>& prompt,
    int prompt_len,
    int block_size,
    int request_id,
    double* wall_ms_out,
    double* gpu_ms_out,
    int* next_token_out,
    int* used_blocks_out
) {
    int* d_ids = copy_prompt_to_device(prompt);

    std::vector<PagedKVCache> layer_kv =
        make_layer_kv(
            block_size,
            D_MODEL,
            request_id
        );

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    CUDA_CHECK_LOCAL(cudaEventCreate(&gpu_start));
    CUDA_CHECK_LOCAL(cudaEventCreate(&gpu_stop));

    CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

    double wall_start = now_ms();
    CUDA_CHECK_LOCAL(cudaEventRecord(gpu_start));

    int next_token = model.prefill(
        d_ids,
        prompt_len,
        layer_kv
    );

    CUDA_CHECK_LOCAL(cudaEventRecord(gpu_stop));
    CUDA_CHECK_LOCAL(cudaEventSynchronize(gpu_stop));
    double wall_end = now_ms();

    CUDA_CHECK_LOCAL(cudaGetLastError());

    check_layer_kv_consistency(
        layer_kv,
        prompt_len,
        "after prefill"
    );

    float gpu_ms = elapsed_gpu_ms(
        gpu_start,
        gpu_stop
    );

    int used_blocks = count_layer_kv_blocks(layer_kv);

    if (wall_ms_out) {
        *wall_ms_out = wall_end - wall_start;
    }

    if (gpu_ms_out) {
        *gpu_ms_out = (double)gpu_ms;
    }

    if (next_token_out) {
        *next_token_out = next_token;
    }

    if (used_blocks_out) {
        *used_blocks_out = used_blocks;
    }

    free_layer_kv(layer_kv);

    CUDA_CHECK_LOCAL(cudaEventDestroy(gpu_start));
    CUDA_CHECK_LOCAL(cudaEventDestroy(gpu_stop));
    CUDA_CHECK_LOCAL(cudaFree(d_ids));

    CUDA_CHECK_LOCAL(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    int warmup_runs  = 3;
    int measure_runs = 10;
    int block_size   = 16;
    int total_blocks = 8192;

    std::vector<int> prompt_lens = {
        32, 64, 128, 256, 512, 1024
    };

    if (argc >= 2) {
        warmup_runs = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        measure_runs = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        block_size = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        total_blocks = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        prompt_lens.clear();

        for (int i = 5; i < argc; i++) {
            prompt_lens.push_back(std::atoi(argv[i]));
        }
    }

    if (warmup_runs < 0 || measure_runs <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (block_size <= 0 || total_blocks <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (prompt_lens.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    int max_prompt_len = 0;

    for (int len : prompt_lens) {
        if (len <= 0 || len > MAX_SEQ) {
            fprintf(stderr,
                    "Invalid prompt_len=%d. It must be in [1, %d].\n",
                    len,
                    MAX_SEQ);
            return 1;
        }

        max_prompt_len = std::max(max_prompt_len, len);
    }

    int max_blocks_per_request =
        ((max_prompt_len + block_size - 1) / block_size) * N_LAYERS;

    if (total_blocks < max_blocks_per_request) {
        fprintf(stderr,
                "Warning: total_blocks=%d may be too small. "
                "Max prompt_len=%d requires at least %d blocks "
                "for one request with block_size=%d and N_LAYERS=%d.\n",
                total_blocks,
                max_prompt_len,
                max_blocks_per_request,
                block_size,
                N_LAYERS);
    }

    /*
     * BlockAllocator singleton 초기화.
     * 반드시 GPT2ModelWMMA::init 전에 한 번 호출.
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

    printf("[config] warmup_runs=%d measure_runs=%d block_size=%d total_blocks=%d max_prompt_len=%d\n",
           warmup_runs,
           measure_runs,
           block_size,
           total_blocks,
           max_prompt_len);

    printf("prompt_len,blocks_per_request_estimate,used_blocks_mean,wall_ms_mean,wall_ms_p50,wall_ms_p95,gpu_ms_mean,gpu_ms_p50,gpu_ms_p95,next_token_last,free_blocks_after\n");

    int request_id_base = 1000;

    for (int prompt_len : prompt_lens) {
        std::vector<int> prompt = make_prompt(prompt_len);

        int blocks_per_request_estimate =
            ((prompt_len + block_size - 1) / block_size) * N_LAYERS;

        /*
         * Warmup.
         */
        for (int i = 0; i < warmup_runs; i++) {
            double wall_ms = 0.0;
            double gpu_ms = 0.0;
            int next_token = -1;
            int used_blocks = 0;

            run_single_prefill_once(
                model,
                prompt,
                prompt_len,
                block_size,
                request_id_base++,
                &wall_ms,
                &gpu_ms,
                &next_token,
                &used_blocks
            );
        }

        std::vector<double> wall_times;
        std::vector<double> gpu_times;
        std::vector<double> used_blocks_vec;

        wall_times.reserve(measure_runs);
        gpu_times.reserve(measure_runs);
        used_blocks_vec.reserve(measure_runs);

        int next_token_last = -1;

        for (int r = 0; r < measure_runs; r++) {
            double wall_ms = 0.0;
            double gpu_ms = 0.0;
            int next_token = -1;
            int used_blocks = 0;

            run_single_prefill_once(
                model,
                prompt,
                prompt_len,
                block_size,
                request_id_base++,
                &wall_ms,
                &gpu_ms,
                &next_token,
                &used_blocks
            );

            wall_times.push_back(wall_ms);
            gpu_times.push_back(gpu_ms);
            used_blocks_vec.push_back((double)used_blocks);

            next_token_last = next_token;
        }

        int free_blocks_after =
            BlockAllocator::getInstance().get_num_free_blocks();

        printf("%d,%d,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n",
               prompt_len,
               blocks_per_request_estimate,
               mean(used_blocks_vec),
               mean(wall_times),
               percentile(wall_times, 0.50),
               percentile(wall_times, 0.95),
               mean(gpu_times),
               percentile(gpu_times, 0.50),
               percentile(gpu_times, 0.95),
               next_token_last,
               free_blocks_after);
    }

    return 0;
}
