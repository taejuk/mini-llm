#include "model/gpt2_model.cuh"
#include "kernels/embedding.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/flashattention.cuh"
#include "kernels/prefill/append_kv.cuh"
#include "kernels/residual.cuh"


#include "runtime/pagekvcache.h"
#include "runtime/pool.cuh"

namespace mini_llm::model {
namespace C = mini_llm::constants;
namespace Kernel = mini_llm::kernels;
namespace Rt = mini_llm::runtime;
static GPT2Weights load_weights() {
    GPT2Weights W; char path[512];
#define LOAD(ptr, fname, n) \
    snprintf(path, 512, "%s/%s", WEIGHTS_DIR, fname); ptr = load_bin(path, (size_t)(n));
    LOAD(W.wte, "wte.bin", (size_t)C::GPT2_VOCAB_SIZE*C::GPT2_D_MODEL);
    LOAD(W.wpe, "wpe.bin", (size_t)C::MAX_SEQ*C::GPT2_D_MODEL);
    for (int l = 0; l < C::GPT2_N_LAYERS; l++) {
        char nm[64];
        snprintf(nm,64,"ln1_w_%d.bin",l); LOAD(W.ln1_w[l], nm, C::GPT2_D_MODEL);
        snprintf(nm,64,"ln1_b_%d.bin",l); LOAD(W.ln1_b[l], nm, C::GPT2_D_MODEL);
        snprintf(nm,64,"qkv_w_%d.bin",l); LOAD(W.qkv_w[l], nm, (size_t)3*C::GPT2_D_MODEL*C::GPT2_D_MODEL);
        snprintf(nm,64,"qkv_b_%d.bin",l); LOAD(W.qkv_b[l], nm, 3*C::GPT2_D_MODEL);
        snprintf(nm,64,"out_w_%d.bin",l); LOAD(W.out_w[l], nm, (size_t)C::GPT2_D_MODEL*C::GPT2_D_MODEL);
        snprintf(nm,64,"out_b_%d.bin",l); LOAD(W.out_b[l], nm, C::GPT2_D_MODEL);
        snprintf(nm,64,"ln2_w_%d.bin",l); LOAD(W.ln2_w[l], nm, C::GPT2_D_MODEL);
        snprintf(nm,64,"ln2_b_%d.bin",l); LOAD(W.ln2_b[l], nm, C::GPT2_D_MODEL);
        snprintf(nm,64,"fc1_w_%d.bin",l); LOAD(W.fc1_w[l], nm, (size_t)C::GPT2_D_FF*C::GPT2_D_MODEL);
        snprintf(nm,64,"fc1_b_%d.bin",l); LOAD(W.fc1_b[l], nm, C::GPT2_D_FF);
        snprintf(nm,64,"fc2_w_%d.bin",l); LOAD(W.fc2_w[l], nm, (size_t)C::GPT2_D_MODEL*C::GPT2_D_FF);
        snprintf(nm,64,"fc2_b_%d.bin",l); LOAD(W.fc2_b[l], nm, C::GPT2_D_MODEL);
    }
    LOAD(W.ln_f_w, "ln_f_w.bin", C::GPT2_D_MODEL);
    LOAD(W.ln_f_b, "ln_f_b.bin", C::GPT2_D_MODEL);
#undef LOAD
    return W;
}

GPT2Model::GPT2Model() {
    W = load_weights();
    // buf_qkv랑 MAX_BATCH수를 조절하면서 할 것
    cudaMalloc(&buf_x, C::MAX_BATCH_NUM * C::MAX_SEQ * C::GPT2_D_MODEL * sizeof(float));
    cudaMalloc(&buf_ln, C::MAX_BATCH_NUM * C::MAX_SEQ * C::GPT2_D_MODEL * sizeof(float));
    cudaMalloc(&buf_qkv, C::MAX_BATCH_NUM * C::MAX_SEQ * C::GPT2_D_MODEL * 3 * sizeof(float));
    cudaMalloc(&buf_attn_out, C::MAX_BATCH_NUM * C::MAX_SEQ * C::GPT2_D_MODEL * sizeof(float));
    cudaMalloc(&buf_proj, C::MAX_BATCH_NUM * C::MAX_SEQ * C::GPT2_D_MODEL * sizeof(float));
    cudaMalloc(&buf_ff, C::MAX_BATCH_NUM * C::MAX_SEQ * C::GPT2_D_FF * sizeof(float));
    cudaMalloc(&buf_x_last, C::MAX_BATCH_NUM * C::GPT2_D_MODEL);
    cudaMalloc(&buf_logits, C::MAX_BATCH_NUM * C::GPT2_VOCAB_SIZE * sizeof(float));
    cudaMalloc(&d_tokens, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    cudaMalloc(&d_pos, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    cudaMalloc(&d_token_to_block, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    cudaMalloc(&d_token_to_offset, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    h_token_to_block = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_token_to_offset = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_tokens = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_pos = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    int max_blocks = (C::MAX_BATCH_NUM * C::MAX_SEQ) / C::DEFAULT_KV_BLOCK_SIZE + 1;

    cudaMalloc(&d_block_table, max_blocks * sizeof(int));
    pool = mini_llm::runtime::Pool::getInstance().pool_start();

    
}

GPT2Model& GPT2Model::get() {
    static GPT2Model gpt2_model;
    return gpt2_model;
}
// buf_qkv를 채우기위한 table을 만든다.
void GPT2Model::make_tables(std::vector<unique_ptr<Request>>& reqs, int layer) {
    size_t cur = 0;
    for(const auto& req: reqs) {
        int tokens_size = req->prompts.size();
        PagedKVCache& kv = req->layer_kv[layer];
        for(int i = 0; i < tokens_size; i++) {
            int physical_block = kv.physical_block_id(i / C::DEFAULT_KV_BLOCK_SIZE);
            int offset = i % C::DEFAULT_KV_BLOCK_SIZE;
            h_token_to_block[cur] = physical_block;
            h_token_to_offset[cur] = offset;
            cur++;
        }
    }
    cudaMemcpy(d_token_to_block, h_token_to_block, cur * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_token_to_offset, h_token_to_offset, cur * sizeof(int), cudaMemcpyHostToDevice);
}

void GPT2Model::gather_last_tokens(std::vector<unique_ptr<Request>>& reqs) {
    int offset = 0;
    int idx = 0;
    for(const auto& req: reqs) {
        cudaMemcpy(buf_x_last + idx * C::GPT2_D_MODEL, buf_x + (offset + req->prompts.size()-1)*C::GPT2_D_MODEL, C::GPT2_D_MODEL*sizeof(float), cudaMemcpyDeviceToDevice);
        idx++;
        offset += req->prompts.size();
    }
}

void GPT2Model::block_prefill(std::vector<unique_ptr<Request>>& reqs, int seq_len, int layer) {
Kernel::layernorm(buf_x, W.ln1_w[layer], W.ln1_b[layer], buf_ln, seq_len, mini_llm::constants::GPT2_D_MODEL);
        // buf_qkv를 만들어야 한다.
        Kernel::qkv_projection(
            buf_ln,
            W.qkv_w[layer],
            W.qkv_b[layer],
            buf_qkv,
            seq_len
        );
        // buf_qkv를 저장하는 것은 모든 request가 같이하는게 낫다.
        make_tables(reqs, layer);
        Kernel::append_prefill_kv(buf_qkv, pool, d_token_to_block, d_token_to_offset, seq_len);
        // attention 호출
        int before_tokens = 0;   
        float scale = 1.0f / sqrtf(static_cast<float>(mini_llm::constants::GPT2_D_HEAD));
        for(const auto& req: reqs) {
            int buf_qkv_offset = before_tokens * 3 * C::GPT2_D_MODEL;
            int buf_O_offset = before_tokens * C::GPT2_D_MODEL;
            Kernel::flashattention_prefill(buf_qkv + buf_qkv_offset, buf_attn_out + buf_O_offset, req->prompts.size(), scale);
        }
        // output projection을 한다.
        Kernel::linear(
            buf_attn_out,
            W.out_w[layer],
            W.out_b[layer],
            buf_proj,
            seq_len,
            mini_llm::constants::GPT2_D_MODEL,
            mini_llm::constants::GPT2_D_MODEL
        );

        Kernel::residual_add(buf_x, buf_proj, (size_t)seq_len * mini_llm::constants::GPT2_D_MODEL);
        Kernel::layernorm(buf_x, W.ln2_w[layer], W.ln2_b[layer], buf_ln ,seq_len, mini_llm::constants::GPT2_D_MODEL);
        Kernel::linear(buf_ln, W.fc1_w[layer], W.fc1_b[layer], buf_ff, seq_len, mini_llm::constants::GPT2_D_MODEL, mini_llm::constants::GPT2_D_FF);
        Kernel::gelu(buf_ff, seq_len * mini_llm::constants::GPT2_D_FF);
        Kernel::linear(buf_ff, W.fc2_w[layer], W.fc2_b[layer], buf_proj, seq_len, mini_llm::constants::GPT2_D_FF, mini_llm::constants::GPT2_D_MODEL);
        Kernel::residual_add(buf_x, buf_proj, seq_len * mini_llm::constants::GPT2_D_MODEL);
}

static std::vector<int> argmax_cpu(const float* v, int row, int col) {
    std::vector<int> result;
    for(int r = 0; r < row; r++) {
        int best = 0;
        for(int i = 1; i < col; i++)
            if(v[r * row + best] < v[r * row + i]) best = i;
        result.push_back(best);
    }
    return result;
}

std::vector<Rt::Response> GPT2Model::prefill(std::vector<unique_ptr<Rt::Request>>& reqs) {
    size_t seq_len = 0;
    for(const auto& req: reqs) seq_len += req->prompts.size();
    int pos = 0;
    for(const auto& req: reqs) {
        for(int i = 0; i < req->prompts.size(); i++) {
            h_tokens[pos] = req->prompts[i];
            h_pos[pos] = i;
            pos++;
        }
    }
    int size = seq_len*sizeof(int);
    cudaMemcpy(d_tokens, h_tokens, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
    
    // 1. embedding 호출
    Kernel::embedding_lookup(d_tokens, d_pos, W.wte, W.wpe, buf_x, seq_len);
    
    // block prefill을 여기서 해야한다.
    for(int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        block_prefill(reqs, seq_len, layer);
    }
    // last block을 가져와서 합쳐야 한다.
    gather_last_tokens(reqs);
    Kernel::layernorm(buf_last_x, W.ln_f_w, W.ln_f_b, buf_ln,reqs.size(), mini_llm::constants::GPT2_D_MODEL);
    Kernel::launch_gemm(mini_llm::constants::GPT2_VOCAB_SIZE, reqs.size(), mini_llm::constants::GPT2_D_MODEL, 1.0f, W.wte, buf_ln, 0.0f, buf_logits);
    float* h_logits = (float*)malloc(mini_llm::constants::GPT2_VOCAB_SIZE * reqs.size() * sizeof(float));
    cudaMemcpy(h_logits, buf_logits, mini_llm::constants::GPT2_VOCAB_SIZE * reqs.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<int> output_tokens = argmax_cpu(h_logits, reqs.size() ,mini_llm::constants::GPT2_VOCAB_SIZE);
    std::vector<Rt::Response> result;
    for(int i = 0; i < reqs.size(); i++) {
        bool done = reqs[i]->max_new_tokens <= 1;
        result.emplace_back(reqs[i]->request_id, output_tokens[i], done);
    }
    
    return result;
}


}