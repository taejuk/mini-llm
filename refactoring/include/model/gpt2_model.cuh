#pragma once
#include <cuda_runtime.h>

#include "runtime/pagekvcache.h"
#include "constants.h"

namespace mini_llm::model {
struct GPT2Weigts {
    float* wte;
    float* wpe;
    float* ln1_w[mini_llm::constants::GPT2_N_LAYERS];
    float* ln1_b[mini_llm::constants::GPT2_N_LAYERS];
    float* qkv_w[mini_llm::constants::GPT2_N_LAYERS];
    float* qkv_b[mini_llm::constants::GPT2_N_LAYERS];
    float* out_w[mini_llm::constants::GPT2_N_LAYERS];
    float* out_b[mini_llm::constants::GPT2_N_LAYERS];
    float* ln2_w[mini_llm::constants::GPT2_N_LAYERS];
    float* ln2_b[mini_llm::constants::GPT2_N_LAYERS];
    float* fc1_w[mini_llm::constants::GPT2_N_LAYERS]; 
    float* fc1_b[mini_llm::constants::GPT2_N_LAYERS];
    float* fc2_w[mini_llm::constants::GPT2_N_LAYERS];
    float* fc2_b[mini_llm::constants::GPT2_N_LAYERS];
    float* ln_f_w;
    float* ln_f_b;
};

class GPT2Model {
private:
    GPT2Weigts W;
    float* buf_x;
    float* buf_ln;
    float* buf_qkv;
    float* buf_attn;
    float* buf_ff;
    float* buf_logits;

    int* d_block_table;
    int* d_tokens;
    int* d_pos;
    GPT2Model();

public:
    static GPT2Model& get();

    GPT2Model(const GPT2Model&) = delete;
    GPT2Model& operator=(const GPT2Model&) = delete;

    std::vector<Response> prefill(std::vector<unique_ptr<Request>>& reqs);
    std::vector<Response> decode(std::vector<unique_ptr<Request>>& reqs);
};


}