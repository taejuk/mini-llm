#include "model/gpt2_wmma.cuh"
#include "model/gpt2_common.cuh"
#include "vllm/allocator.cuh"
#include "vllm/kv_cache.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

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
        // 실제 문장 기반 비교가 필요하면 이 배열을 tokenizer 결과로 교체하면 됨.
        ids[i] = 100 + (i % 1000);
    }

    return ids;
}

static int* copy_prompt_to_device(const std::vector<int>& ids) {
    int* d_ids = nullptr;

    CUDA_CHECK_LOCAL(cudaMalloc(&d_ids, ids.size() * sizeof(int)));

    CUDA_CHECK_LOCAL(cudaMemcpy(
        d_ids,
        ids.data(),
        ids.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    return d_ids;
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
        prog, prog);
}

static void run_warmup(
    GPT2ModelWMMA& model,
    const std::vector<int>& prompt,
    int decode_steps,
    int block_size,
    int request_id
) {
    int* d_ids = copy_prompt_to_device(prompt);
    PagedKVCache kv(block_size, D_MODEL, request_id);

    int token = model.prefill(d_ids, (int)prompt.size(), kv);

    for (int i = 0; i < decode_steps; i++) {
        token = model.decode_step(token, kv);
    }

    CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

    kv.free_all();
    CUDA_CHECK_LOCAL(cudaFree(d_ids));
}

int main(int argc, char** argv) {
    int prompt_len   = 128;
    int decode_steps = 32;
    int warmup_steps = 5;
    int block_size   = 16;
    int total_blocks = 4096;

    if (argc >= 2) prompt_len   = std::atoi(argv[1]);
    if (argc >= 3) decode_steps = std::atoi(argv[2]);
    if (argc >= 4) warmup_steps = std::atoi(argv[3]);
    if (argc >= 5) block_size   = std::atoi(argv[4]);
    if (argc >= 6) total_blocks = std::atoi(argv[5]);

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
        fprintf(stderr, "Invalid decode_steps=%d.\n", decode_steps);
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

    // BlockAllocator singleton 초기화.
    // GPT2ModelWMMA 내부 decode attention이 이 allocator pool을 사용한다.
    BlockAllocator::getInstance(
        total_blocks,
        block_size,
        D_MODEL
    );

    GPT2ModelWMMA::init(WEIGHTS_DIR, block_size);
    GPT2ModelWMMA& model = GPT2ModelWMMA::get();

    std::vector<int> prompt = make_prompt(prompt_len);

    // Warmup
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
    PagedKVCache kv(block_size, D_MODEL, 1);

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    CUDA_CHECK_LOCAL(cudaEventCreate(&gpu_start));
    CUDA_CHECK_LOCAL(cudaEventCreate(&gpu_stop));

    printf("phase,token_index,seq_len_before,input_token,output_token,wall_ms,gpu_ms\n");

    // 1) Prefill 측정
    CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

    double wall_start = now_ms();
    CUDA_CHECK_LOCAL(cudaEventRecord(gpu_start));

    int next_token = model.prefill(
        d_ids,
        prompt_len,
        kv
    );

    CUDA_CHECK_LOCAL(cudaEventRecord(gpu_stop));
    CUDA_CHECK_LOCAL(cudaEventSynchronize(gpu_stop));
    double wall_end = now_ms();

    float gpu_ms = elapsed_gpu_ms(gpu_start, gpu_stop);
    double wall_ms = wall_end - wall_start;

    printf("prefill,%d,%d,%d,%d,%.6f,%.6f\n",
           0,
           0,
           prompt.empty() ? -1 : prompt.back(),
           next_token,
           wall_ms,
           gpu_ms);

    // 2) Decode token별 측정
    int input_token = next_token;

    for (int t = 0; t < decode_steps; t++) {
        int seq_len_before = kv.get_num_tokens();

        CUDA_CHECK_LOCAL(cudaDeviceSynchronize());

        wall_start = now_ms();
        CUDA_CHECK_LOCAL(cudaEventRecord(gpu_start));

        int out_token = model.decode_step(
            input_token,
            kv
        );

        CUDA_CHECK_LOCAL(cudaEventRecord(gpu_stop));
        CUDA_CHECK_LOCAL(cudaEventSynchronize(gpu_stop));
        wall_end = now_ms();

        gpu_ms = elapsed_gpu_ms(gpu_start, gpu_stop);
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

    kv.free_all();

    CUDA_CHECK_LOCAL(cudaEventDestroy(gpu_start));
    CUDA_CHECK_LOCAL(cudaEventDestroy(gpu_stop));
    CUDA_CHECK_LOCAL(cudaFree(d_ids));

    return 0;
}
