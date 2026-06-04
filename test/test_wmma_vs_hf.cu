#include "model/gpt2_wmma.cuh"
#include "vllm/kv_cache.cuh"
#ifdef ENABLE_TENSOR_DUMP
#include "debug/tensor_dumper.h"
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            std::exit(1);                                                   \
        }                                                                  \
    } while (0)



int main(int argc, char** argv) {
    constexpr int BLOCK_SIZE = 16;
    constexpr int TOTAL_BLOCKS = 128;
    const char* weight_dir = WEIGHTS_DIR;

    if (argc >= 2) {
        weight_dir = argv[1];
    }

    BlockAllocator::getInstance(TOTAL_BLOCKS, BLOCK_SIZE, D_MODEL);
    std::cout << "[main] allocator ready: " << TOTAL_BLOCKS << " blocks\n";

    GPT2ModelWMMA::init(weight_dir, BLOCK_SIZE);
    auto& model = GPT2ModelWMMA::get();

#ifdef ENABLE_TENSOR_DUMP
    TensorDumpConfig cfg;
    cfg.enabled = true;
    cfg.output_dir = "mini_ref";

    // 지금은 embedding만 확인
    cfg.allowlist = {
        "wmma_embedding_out"
    };

    TensorDumper dumper(cfg);
    model.set_tensor_dumper(&dumper);
#endif

    // Hugging Face GPT-2 tokenizer 기준:
    // "Hello, my name is" -> [15496, 11, 616, 1438, 318]
    // 나중에는 hf_ref/input_ids.npy에서 읽어오도록 바꾸면 됨.
    std::vector<int> h_token_ids = {
        15496, 11, 616, 1438, 318
    };

    int prompt_len = static_cast<int>(h_token_ids.size());

    int* d_token_ids = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_token_ids,
        static_cast<size_t>(prompt_len) * sizeof(int)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_token_ids,
        h_token_ids.data(),
        static_cast<size_t>(prompt_len) * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    // GPT2ModelWMMA::prefill은 layer별 KV cache를 요구함
    std::vector<PagedKVCache> layer_kv;
    layer_kv.reserve(N_LAYERS);

    for (int layer = 0; layer < N_LAYERS; layer++) {
        layer_kv.emplace_back(
            BLOCK_SIZE,
            D_MODEL,
            layer
        );
    }

    int next_tok = model.prefill(
        d_token_ids,
        prompt_len,
        layer_kv
    );

    printf("next token = %d\n", next_tok);

    CUDA_CHECK(cudaFree(d_token_ids));

    return 0;
}
