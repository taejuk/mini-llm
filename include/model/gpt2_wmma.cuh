#pragma once

#include "model/gpt2_common.cuh"
#include "kernel/wmma_gemm.cuh"
#include "vllm/kv_cache.cuh"
#include "scheduler/request.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

#ifdef ENABLE_TENSOR_DUMP
#include "debug/tensor_dumper.h"
#endif

#define MAX_BATCH 32

struct GPT2WeightsWMMA {
    float* wte;
    float* wpe;

    float* ln1_w[N_LAYERS];
    float* ln1_b[N_LAYERS];
    float* ln2_w[N_LAYERS];
    float* ln2_b[N_LAYERS];
    float* ln_f_w;
    float* ln_f_b;

    float* qkv_b[N_LAYERS];
    float* out_b[N_LAYERS];
    float* fc1_b[N_LAYERS];
    float* fc2_b[N_LAYERS];

    __half* qkv_w[N_LAYERS];
    __half* out_w[N_LAYERS];
    __half* fc1_w[N_LAYERS];
    __half* fc2_w[N_LAYERS];
};

class GPT2ModelWMMA {
public:
    static void           init(const char* weight_dir, int blk_size);
    static GPT2ModelWMMA& get();

    GPT2ModelWMMA(const GPT2ModelWMMA&)            = delete;
    GPT2ModelWMMA& operator=(const GPT2ModelWMMA&) = delete;

#ifdef ENABLE_TENSOR_DUMP
    void set_tensor_dumper(TensorDumper* dumper) {
        dumper_ = dumper;
    }
#endif

    int prefill(const int* d_token_ids, int prompt_len, std::vector<PagedKVCache>& layer_kv);
    int decode_step(int token_id, std::vector<PagedKVCache>& layer_kv);

    std::vector<int> batch_decode(const std::vector<Request*>& reqs);

    float* get_logits() const { return buf_logits; }

private:
    GPT2WeightsWMMA W;
    cublasHandle_t  handle;

    float*  buf_x;
    float*  buf_ln;
    float*  buf_qkv;
    __half* buf_xh;
    float*  buf_S;
    float*  buf_O;
    float*  buf_attn;
    float*  buf_ff;
    float*  buf_logits;

    int* d_block_table;

    int*   d_batch_block_tables;
    int*   d_seq_lens;
    float* buf_batch_logits;

    int block_size;

#ifdef ENABLE_TENSOR_DUMP
    TensorDumper* dumper_ = nullptr; // non-owning
#endif

    GPT2ModelWMMA(const char* weight_dir, int blk_size);
};