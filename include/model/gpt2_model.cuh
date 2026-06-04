#pragma once

#include "vllm/kv_cache.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef ENABLE_TENSOR_DUMP
#include "debug/tensor_dumper.h"
#endif

#define D_MODEL   768
#define N_HEADS   12
#define D_HEAD    64
#define N_LAYERS  12
#define D_FF      3072
#define VOCAB     50257
#define MAX_SEQ   1024

struct GPT2Weights {
  float* wte;              // [VOCAB, D_MODEL]
  float* wpe;              // [MAX_SEQ, D_MODEL]
  float* ln1_w[N_LAYERS];  // [D_MODEL]
  float* ln1_b[N_LAYERS];
  float* qkv_w[N_LAYERS];  // [3*D_MODEL, D_MODEL]
  float* qkv_b[N_LAYERS];  // [3*D_MODEL]
  float* out_w[N_LAYERS];  // [D_MODEL, D_MODEL]
  float* out_b[N_LAYERS];  // [D_MODEL]
  float* ln2_w[N_LAYERS];
  float* ln2_b[N_LAYERS];
  float* fc1_w[N_LAYERS];  // [D_FF, D_MODEL]
  float* fc1_b[N_LAYERS];  // [D_FF]
  float* fc2_w[N_LAYERS];  // [D_MODEL, D_FF]
  float* fc2_b[N_LAYERS];  // [D_MODEL]
  float* ln_f_w;
  float* ln_f_b;
};


class GPT2Model {
private:
  GPT2Weights W;
  cublasHandle_t handle;


  float* buf_x;
  float* buf_ln;
  float* buf_qkv;
  float* buf_S;
  float* buf_O;
  float* buf_attn;
  float* buf_ff;
  float* buf_logits;

  int* d_block_table;
  int block_size;
  GPT2Model(const char* weight_dir, int blk_size);


public:

  static void init(const char* weight_dir, int blk_size);
  static GPT2Model& get();

  GPT2Model(const GPT2Model&) = delete;
  GPT2Model& operator=(const GPT2Model&) = delete;

  int prefill(const int* d_token_ids, int prompt_len, PagedKVCache& kv);
  int decode_step(int token_id, PagedKVCache& kv);
  float* get_logits() const { return buf_logits; }
};
