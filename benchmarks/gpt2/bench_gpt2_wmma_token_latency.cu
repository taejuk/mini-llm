#include "model/gpt2_wmma.cuh"
#include "model/gpt2_common.cuh"
#include "vllm/allocator.cuh"
#include "vllm/kv_cache.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
        // GPT-2 vocab 범위 내의 임의 token id.
        // 실제 문장 기반 비교가 필요하면 이 배열을 tokenizer 결과로 교체하면 된다.
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

static void free_layer_kv(
    std::vector<PagedKVCache>& layer_kv
) {
    for (auto& kv : layer_kv) {
        kv.free_all();
    }
}

static int get_seq_len_from_layer_kv(
    const std::vector<PagedKVCache>& layer_kv
) {
    if (layer_kv.empty()) {
        return 0;
    }

    return layer_kv[0].get_num_tokens();
}

static void check_layer_kv_consistency(
    const std::vector<PagedKVCache>& layer_kv,
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

    int base_len = layer_kv[0].get_num_tokens();

    for (int l = 1; l < N_LAYERS; l++) {
        int len = layer_kv[l].get_num_tokens();

        if (len != base_len) {
            fprintf(stderr,
                    "%s: inconsistent layer KV length: "
                    "layer 0 len=%d, layer %d len=%d\n",
                    where,
                    base_len,
                    l,
                    len);
            std::exit(1);
        }
    }
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [prompt_len] [decode_steps] [warmup_steps] [block_size] [total_blocks]\n"
        "\n"
        "Example:\n"
        "  %s 128 32 5 16 4096\n"
        "\n"
        "Arguments:\n"
        "  prompt_len    : prompt token length, default 128\n"
        "  decode_steps  : number of generated tokens to benchmark, default 32\n"
        "  warmup_steps  : warmup runs before measurement, default 5\n"
        "  block_size    : paged KV block size, default 16\n"
        "  total_blocks  : total physical KV blocks, default 4096\n",
        prog,
        prog);
}

static void run_warmup(
    GPT2ModelWMMA& model,
    const std::vector<int>& prompt,
    int decode_steps,
    int block_size,
    int request_id
) {
    int* d_ids = copy_prompt_to_device(prompt);

    std::vector<PagedKVCache> layer_kv =
        make_layer_kv(
            block_size,
            D_MODEL,
            request_id
        );

    int token = model.prefill(
        d_ids,
        (int)prompt.size(),
        layer_kv
    );

    check_layer_kv_consistency(
        layer_kv,
        "warmup after prefill"
    );

    for (int i = 0; i < decode_steps; i++) {
        token = model.decode_step(
            token,
            layer_kv
        );

        check_layer_kv_consistency(
            layer_kv,
            "warmup after decode_step"
        );
    }

    CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

    free_layer_kv(layer_kv);
    CUDA_CHECK_LOCAL(cudaFree(d_ids));
}

int main(int argc, char** argv) {
    int prompt_len   = 128;
    int decode_steps = 32;
    int warmup_steps = 5;
    int block_size   = 16;
    int total_blocks = 4096;

    if (argc >= 2) {
        prompt_len = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        decode_steps = std::atoi(argv[2]);
    }

    if (argc >= 4) {
        warmup_steps = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        block_size = std::atoi(argv[4]);
    }

    if (argc >= 6) {
        total_blocks = std::atoi(argv[5]);
    }

    if (argc > 6) {
        print_usage(argv[0]);
        return 1;
    }

    if (prompt_len <= 0 || prompt_len > MAX_SEQ) {
        fprintf(stderr,
                "Invalid prompt_len=%d. It must be in [1, %d].\n",
                prompt_len,
                MAX_SEQ);
        return 1;
    }

    if (decode_steps < 0) {
        fprintf(stderr,
                "Invalid decode_steps=%d.\n",
                decode_steps);
        return 1;
    }

    if (prompt_len + decode_steps > MAX_SEQ) {
        fprintf(stderr,
                "prompt_len + decode_steps exceeds MAX_SEQ: %d + %d > %d\n",
                prompt_len,
                decode_steps,
                MAX_SEQ);
        return 1;
    }

    if (block_size <= 0 || total_blocks <= 0) {
        fprintf(stderr,
                "Invalid block_size=%d or total_blocks=%d.\n",
                block_size,
                total_blocks);
        return 1;
    }

    int blocks_per_layer =
        (prompt_len + decode_steps + block_size - 1) / block_size;

    int min_blocks_for_one_request =
        blocks_per_layer * N_LAYERS;

    if (total_blocks < min_blocks_for_one_request) {
        fprintf(stderr,
                "Warning: total_blocks=%d may be too small. "
                "At least %d blocks are needed for one request "
                "with prompt_len=%d, decode_steps=%d, block_size=%d, N_LAYERS=%d.\n",
                total_blocks,
                min_blocks_for_one_request,
                prompt_len,
                decode_steps,
                block_size,
                N_LAYERS);
    }

    /*
     * BlockAllocator singleton 초기화.
     *
     * layer별 KV 구조에서는 request 하나가 N_LAYERS개의 PagedKVCache를 가진다.
     * 따라서 total_blocks는 적어도:
     *
     *   ceil((prompt_len + decode_steps) / block_size) * N_LAYERS
     *
     * 이상이어야 한다.
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

    std::vector<int> prompt = make_prompt(prompt_len);

    /*
     * Warmup.
     *
     * CUDA kernel launch, cuBLAS initialization, cache effects 등을 제외하기 위해
     * 실제 측정 전에 몇 번 실행한다.
     */
    for (int i = 0; i < warmup_steps; i++) {
        run_warmup(
            model,
            prompt,
            decode_steps,
            block_size,
            100000 + i
        );
    }

    int* d_ids = copy_prompt_to_device(prompt);

    std::vector<PagedKVCache> layer_kv =
        make_layer_kv(
            block_size,
            D_MODEL,
            1
        );

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    CUDA_CHECK_LOCAL(cudaEventCreate(&gpu_start));
    CUDA_CHECK_LOCAL(cudaEventCreate(&gpu_stop));

    printf("phase,token_index,seq_len_before,input_token,output_token,wall_ms,gpu_ms\n");

    /*
     * 1. Prefill 측정.
     *
     * token_index = 0:
     *   prompt 전체를 한 번에 처리하는 prefill latency.
     *
     * seq_len_before = 0:
     *   prefill 전에는 KV cache에 token이 없다는 의미.
     */
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
        "measurement after prefill"
    );

    float gpu_ms = elapsed_gpu_ms(
        gpu_start,
        gpu_stop
    );

    double wall_ms = wall_end - wall_start;

    printf("prefill,%d,%d,%d,%d,%.6f,%.6f\n",
           0,
           0,
           prompt.empty() ? -1 : prompt.back(),
           next_token,
           wall_ms,
           gpu_ms);

    /*
     * 2. Decode token별 측정.
     *
     * 각 decode_step은 현재 input_token 하나를 받아 next token 하나를 생성한다.
     * seq_len_before는 decode_step을 호출하기 직전의 KV length다.
     */
    int input_token = next_token;

    for (int t = 0; t < decode_steps; t++) {
        check_layer_kv_consistency(
            layer_kv,
            "before measured decode_step"
        );

        int seq_len_before =
            get_seq_len_from_layer_kv(layer_kv);

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        wall_start = now_ms();
        CUDA_CHECK_LOCAL(cudaEventRecord(gpu_start));

        int out_token = model.decode_step(
            input_token,
            layer_kv
        );

        CUDA_CHECK_LOCAL(cudaEventRecord(gpu_stop));
        CUDA_CHECK_LOCAL(cudaEventSynchronize(gpu_stop));
        wall_end = now_ms();

        CUDA_CHECK_LOCAL(cudaGetLastError());

        check_layer_kv_consistency(
            layer_kv,
            "after measured decode_step"
        );

        gpu_ms = elapsed_gpu_ms(
            gpu_start,
            gpu_stop
        );

        wall_ms = wall_end - wall_start;

        printf("decode,%d,%d,%d,%d,%.6f,%.6f\n",
               t + 1,
               seq_len_before,
               input_token,
               out_token,
               wall_ms,
               gpu_ms);

        input_token = out_token;
    }

    free_layer_kv(layer_kv);

    CUDA_CHECK_LOCAL(cudaEventDestroy(gpu_start));
    CUDA_CHECK_LOCAL(cudaEventDestroy(gpu_stop));
    CUDA_CHECK_LOCAL(cudaFree(d_ids));

    return 0;
}
