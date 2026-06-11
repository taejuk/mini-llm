#pragma once
#include <cuda_runtime.h>

#include "runtime/pagekvcache.h"
#include "runtime/block_manager.h"
#include "constants.h"
#include "runtime/request.h"
#include "runtime/response.h"

namespace mini_llm::model {
namespace Rt = mini_llm::runtime;
struct GPT2Weights {
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
    GPT2Weights W;
    float* buf_x;
    float* buf_ln;
    float* buf_qkv;
    float* buf_attn_out;
    float* buf_proj;
    float* buf_ff;
    float* buf_x_last;
    
    float* buf_logits;

    int* d_tokens;
    int* d_pos;
    int* d_block_table;
    int* d_token_to_block;
    int* d_token_to_offset;

    int* h_token_to_block;
    int* h_token_to_offset;
    int* h_tokens;
    int* h_pos;
    float* h_logits;
    
    float* pool;
    Rt::BlockManager block_manager;
    GPT2Model();
    void make_tables(std::vector<std::unique_ptr<Rt::Request>>& reqs, int layer);
    void block_prefill(std::vector<std::unique_ptr<Rt::Request>>& reqs, int seq_len, int layer);
    void gather_last_tokens(std::vector<std::unique_ptr<Rt::Request>>& reqs);
public:
    static GPT2Model& get();

    GPT2Model(const GPT2Model&) = delete;
    GPT2Model& operator=(const GPT2Model&) = delete;

    std::vector<Rt::Response> prefill(std::vector<unique_ptr<Rt::Request>>& reqs);
    std::vector<Rt::Response> decode(std::vector<unique_ptr<Rt::Request>>& reqs);
};


}