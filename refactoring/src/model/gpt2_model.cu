#include "model/gpt2_model.cuh"
#include "kernels/embedding.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/flashattention.cuh"
#include "kernels/prefill/append_kv.cuh"
#include "kernels/residual.cuh"
#include "kernels/decode/paged_attention.cuh"

#include "runtime/pagekvcache.h"
#include "runtime/pool.cuh"
#include "runtime/block_manager.h"

#include <iostream>
#include <cstdlib>

namespace mini_llm::model {
namespace C = mini_llm::constants;
namespace Kernel = mini_llm::kernels;
namespace Rt = mini_llm::runtime;

__global__ void transpose_wte_kernel(
    const float* __restrict__ wte,     // [VOCAB_SIZE, D_MODEL]
    float* __restrict__ wte_t,         // [D_MODEL, VOCAB_SIZE]
    int vocab_size,
    int d_model
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x; // hidden dim
    int v = blockIdx.y * blockDim.y + threadIdx.y; // vocab index

    if (d < d_model && v < vocab_size) {
        wte_t[static_cast<size_t>(d) * vocab_size + v] =
            wte[static_cast<size_t>(v) * d_model + d];
    }
}

static float* make_wte_t_from_wte(const float* wte) {
    float* wte_t = nullptr;

    size_t n =
        static_cast<size_t>(C::GPT2_VOCAB_SIZE) *
        static_cast<size_t>(C::GPT2_D_MODEL);

    cudaMalloc(&wte_t, n * sizeof(float));

    dim3 block(16, 16);
    dim3 grid(
        (C::GPT2_D_MODEL + block.x - 1) / block.x,
        (C::GPT2_VOCAB_SIZE + block.y - 1) / block.y
    );

    transpose_wte_kernel<<<grid, block>>>(
        wte,
        wte_t,
        C::GPT2_VOCAB_SIZE,
        C::GPT2_D_MODEL
    );

    cudaGetLastError();
    cudaDeviceSynchronize();

    return wte_t;
}

static GPT2Weights load_weights() {
    GPT2Weights W; char path[512];
#define LOAD(ptr, fname, n) \
    snprintf(path, 512, "%s/%s", WEIGHTS_DIR, fname); ptr = load_bin(path, (size_t)(n));
    LOAD(W.wte, "wte.bin", (size_t)C::GPT2_VOCAB_SIZE*C::GPT2_D_MODEL);
    LOAD(W.wpe, "wpe.bin", (size_t)C::MAX_SEQ*C::GPT2_D_MODEL);
    W.wte_t = make_wte_t_from_wte(W.wte);
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
    cudaMalloc(&buf_x_last, C::MAX_BATCH_NUM * C::GPT2_D_MODEL * sizeof(float));
    cudaMalloc(&buf_logits, C::MAX_BATCH_NUM * C::GPT2_VOCAB_SIZE * sizeof(float));
    cudaMalloc(&d_tokens, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    cudaMalloc(&d_pos, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    cudaMalloc(&d_token_to_block, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    cudaMalloc(&d_token_to_offset, C::MAX_BATCH_NUM * C::MAX_SEQ * sizeof(int));
    h_token_to_block = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_token_to_offset = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_tokens = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_pos = new int[C::MAX_BATCH_NUM * C::MAX_SEQ];
    h_logits = new float[C::GPT2_VOCAB_SIZE * C::MAX_BATCH_NUM];
    int max_blocks = (C::MAX_BATCH_NUM * C::MAX_SEQ) / C::DEFAULT_KV_BLOCK_SIZE + 1;

    cudaMalloc(&d_block_table, max_blocks * sizeof(int));
    pool = mini_llm::runtime::Pool::getInstance().pool_start();
}

GPT2Model& GPT2Model::get() {
    static GPT2Model gpt2_model;
    return gpt2_model;
}

void GPT2Model::make_tables(
    std::vector<std::unique_ptr<Rt::Request>>& reqs,
    int layer
) {
    size_t cur = 0;

    for (const auto& req : reqs) {
        int tokens_size = req->prompts_len;
        Rt::PagedKVCache& kv = req->layer_kv[layer];

        for (int i = 0; i < tokens_size; i++) {
            int physical_block =
                kv.physical_block_id(i / C::DEFAULT_KV_BLOCK_SIZE);
            int offset = i % C::DEFAULT_KV_BLOCK_SIZE;

            h_token_to_block[cur] = physical_block;
            h_token_to_offset[cur] = offset;
            cur++;
        }
    }

    cudaMemcpy(
        d_token_to_block,
        h_token_to_block,
        cur * sizeof(int),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_token_to_offset,
        h_token_to_offset,
        cur * sizeof(int),
        cudaMemcpyHostToDevice
    );
}

void GPT2Model::gather_last_tokens(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    int offset = 0;
    int idx = 0;

    for (const auto& req : reqs) {
        int last_prompt_idx = req->prompts_len - 1;

        cudaMemcpy(
            buf_x_last + idx * C::GPT2_D_MODEL,
            buf_x + (offset + last_prompt_idx) * C::GPT2_D_MODEL,
            C::GPT2_D_MODEL * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        idx++;
        offset += req->prompts_len;
    }
}

void GPT2Model::block_prefill(std::vector<std::unique_ptr<Rt::Request>>& reqs, int seq_len, int layer) {
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
            Kernel::flashattention_prefill(buf_qkv + buf_qkv_offset, buf_attn_out + buf_O_offset, req->prompts_len, scale);
            before_tokens += req->prompts_len;
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
            if(v[r * col + best] < v[r * col + i]) best = i;
        result.push_back(best);
    }
    return result;
}

std::vector<Rt::Response> GPT2Model::prefill(std::vector<std::unique_ptr<Rt::Request>>& reqs) {
    size_t seq_len = 0;

    for (const auto& req : reqs) {
        seq_len += req->prompts_len;
    }

    int pos = 0;

    for (const auto& req : reqs) {
        for (int i = 0; i < req->prompts_len; i++) {
            h_tokens[pos] = req->tokens[i];
            h_pos[pos] = i;
            pos++;
        }
    }

    int size = static_cast<int>(seq_len * sizeof(int));

    cudaMemcpy(d_tokens, h_tokens, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos, size, cudaMemcpyHostToDevice);
    // 1. embedding 호출
    Kernel::embedding_lookup(d_tokens, d_pos, W.wte, W.wpe, buf_x, seq_len);
    
    // block prefill을 여기서 해야한다.
    for(int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        // layer마다 block을 직접 할당한다.
        block_prefill(reqs, seq_len, layer);
    }
    // last block을 가져와서 합쳐야 한다.
    gather_last_tokens(reqs);
    Kernel::layernorm(buf_x_last, W.ln_f_w, W.ln_f_b, buf_ln,reqs.size(), mini_llm::constants::GPT2_D_MODEL);
    Kernel::launch_gemm(
        reqs.size(),              // M = batch
        C::GPT2_VOCAB_SIZE,       // N = vocab
        C::GPT2_D_MODEL,          // K = hidden
        1.0f,
        buf_ln,                   // [B, D]
        W.wte_t,                  // [D, V]
        0.0f,
        buf_logits                // [B, V]
    );
    cudaMemcpy(h_logits, buf_logits, mini_llm::constants::GPT2_VOCAB_SIZE * reqs.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<int> output_tokens = argmax_cpu(h_logits, reqs.size() ,mini_llm::constants::GPT2_VOCAB_SIZE);
    std::vector<Rt::Response> result;
    for (int i = 0; i < reqs.size(); i++) {
        bool done = output_tokens[i] == C::GPT2_EOS_TOKEN_ID;
        result.emplace_back(reqs[i]->request_id, output_tokens[i], done);
    }
    return result;
}

std::vector<Rt::Response> GPT2Model::decode(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    int batch_size = static_cast<int>(reqs.size());

    if (batch_size == 0) {
        return {};
    }

    // ------------------------------------------------------------
    // 1. 각 request의 마지막 token을 이번 decode input으로 사용
    // ------------------------------------------------------------
    for (int i = 0; i < batch_size; i++) {
        h_tokens[i] = reqs[i]->tokens.back();

        // tokens.back()의 position은 tokens.size() - 1
        h_pos[i] = static_cast<int>(reqs[i]->tokens.size()) - 1;
    }

    int token_bytes = batch_size * sizeof(int);

    CUDA_CHECK(cudaMemcpy(
        d_tokens,
        h_tokens,
        token_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_pos,
        h_pos,
        token_bytes,
        cudaMemcpyHostToDevice
    ));

    // ------------------------------------------------------------
    // 2. Embedding
    // buf_x: [B, D]
    // ------------------------------------------------------------
    Kernel::embedding_lookup(
        d_tokens,
        d_pos,
        W.wte,
        W.wpe,
        buf_x,
        batch_size
    );

    // layer loop 안에서 사용할 device buffer
    int* d_block_offsets = nullptr;
    int* d_num_tokens = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_block_offsets,
        batch_size * sizeof(int)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_num_tokens,
        batch_size * sizeof(int)
    ));

    // ------------------------------------------------------------
    // 3. Transformer layers
    // ------------------------------------------------------------
    for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        // --------------------------------------------------------
        // 3-1. LN1
        // --------------------------------------------------------
        Kernel::layernorm(
            buf_x,
            W.ln1_w[layer],
            W.ln1_b[layer],
            buf_ln,
            batch_size,
            C::GPT2_D_MODEL
        );

        // --------------------------------------------------------
        // 3-2. QKV projection
        // buf_qkv: [B, 3D]
        // row = [Q | K | V]
        // --------------------------------------------------------
        Kernel::qkv_projection(
            buf_ln,
            W.qkv_w[layer],
            W.qkv_b[layer],
            buf_qkv,
            batch_size
        );

        // --------------------------------------------------------
        // 3-3. 현재 decode token의 K/V를 KV pool에 append
        //
        // kv.num_tokens_는 decode 전까지 cache에 들어간 token 수.
        // 현재 token은 그 위치에 저장한다.
        // --------------------------------------------------------
        for (int i = 0; i < batch_size; i++) {
            Rt::PagedKVCache& kv = reqs[i]->layer_kv[layer];

            int cached_tokens = kv.num_tokens_;
            int logical_block = cached_tokens / C::DEFAULT_KV_BLOCK_SIZE;
            int offset_in_block = cached_tokens % C::DEFAULT_KV_BLOCK_SIZE;

            int physical_block = kv.physical_block_id(logical_block);

            h_token_to_block[i] = physical_block;
            h_token_to_offset[i] = offset_in_block;
        }

        CUDA_CHECK(cudaMemcpy(
            d_token_to_block,
            h_token_to_block,
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_token_to_offset,
            h_token_to_offset,
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        // append_prefill_kv는 이름은 prefill이지만,
        // row별 K/V를 pool의 block/offset에 저장하는 역할이라 decode에서도 재사용 가능
        Kernel::append_prefill_kv(
            buf_qkv,
            pool,
            d_token_to_block,
            d_token_to_offset,
            batch_size
        );

        CUDA_CHECK(cudaGetLastError());

        // --------------------------------------------------------
        // 3-4. paged attention용 flattened block table 구성
        // --------------------------------------------------------
        std::vector<int> h_flat_block_table;
        std::vector<int> h_block_offsets(batch_size);
        std::vector<int> h_num_tokens(batch_size);

        h_flat_block_table.reserve(
            static_cast<size_t>(batch_size) *
            ((C::MAX_SEQ + C::DEFAULT_KV_BLOCK_SIZE - 1) /
             C::DEFAULT_KV_BLOCK_SIZE)
        );

        for (int i = 0; i < batch_size; i++) {
            Rt::PagedKVCache& kv = reqs[i]->layer_kv[layer];

            h_block_offsets[i] =
                static_cast<int>(h_flat_block_table.size());

            for (int block_id : kv.block_table_) {
                h_flat_block_table.push_back(block_id);
            }

            // 방금 append한 현재 token까지 attention 대상에 포함
            h_num_tokens[i] = kv.num_tokens_ + 1;
        }

        CUDA_CHECK(cudaMemcpy(
            d_block_table,
            h_flat_block_table.data(),
            h_flat_block_table.size() * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_block_offsets,
            h_block_offsets.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_num_tokens,
            h_num_tokens.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        // --------------------------------------------------------
        // 3-5. Paged decode attention
        // buf_attn_out: [B, D]
        // --------------------------------------------------------
        Kernel::paged_decode_attention(
            buf_qkv,
            d_block_table,
            d_block_offsets,
            d_num_tokens,
            pool,
            buf_attn_out,
            batch_size,
            C::MAX_SEQ
        );

        CUDA_CHECK(cudaGetLastError());

        // --------------------------------------------------------
        // 3-6. Attention output projection + residual
        // --------------------------------------------------------
        Kernel::linear(
            buf_attn_out,
            W.out_w[layer],
            W.out_b[layer],
            buf_proj,
            batch_size,
            C::GPT2_D_MODEL,
            C::GPT2_D_MODEL
        );

        Kernel::residual_add(
            buf_x,
            buf_proj,
            batch_size * C::GPT2_D_MODEL
        );

        // --------------------------------------------------------
        // 3-7. LN2 + FFN + residual
        // --------------------------------------------------------
        Kernel::layernorm(
            buf_x,
            W.ln2_w[layer],
            W.ln2_b[layer],
            buf_ln,
            batch_size,
            C::GPT2_D_MODEL
        );

        Kernel::linear(
            buf_ln,
            W.fc1_w[layer],
            W.fc1_b[layer],
            buf_ff,
            batch_size,
            C::GPT2_D_MODEL,
            C::GPT2_D_FF
        );

        Kernel::gelu(
            buf_ff,
            batch_size * C::GPT2_D_FF
        );

        Kernel::linear(
            buf_ff,
            W.fc2_w[layer],
            W.fc2_b[layer],
            buf_proj,
            batch_size,
            C::GPT2_D_FF,
            C::GPT2_D_MODEL
        );

        Kernel::residual_add(
            buf_x,
            buf_proj,
            batch_size * C::GPT2_D_MODEL
        );
    }

    CUDA_CHECK(cudaFree(d_block_offsets));
    CUDA_CHECK(cudaFree(d_num_tokens));

    // ------------------------------------------------------------
    // 4. Final LN
    // ------------------------------------------------------------
    Kernel::layernorm(
        buf_x,
        W.ln_f_w,
        W.ln_f_b,
        buf_ln,
        batch_size,
        C::GPT2_D_MODEL
    );

    // ------------------------------------------------------------
    // 5. Vocab projection
    // logits = hidden @ W.wte_t
    // ------------------------------------------------------------
    Kernel::launch_gemm(
        batch_size,              // M = batch
        C::GPT2_VOCAB_SIZE,      // N = vocab
        C::GPT2_D_MODEL,         // K = hidden
        1.0f,
        buf_ln,                  // [B, D]
        W.wte_t,                 // [D, V]
        0.0f,
        buf_logits               // [B, V]
    );

    CUDA_CHECK(cudaMemcpy(
        h_logits,
        buf_logits,
        static_cast<size_t>(batch_size) *
            C::GPT2_VOCAB_SIZE *
            sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::vector<int> output_tokens =
        argmax_cpu(
            h_logits,
            batch_size,
            C::GPT2_VOCAB_SIZE
        );

    // ------------------------------------------------------------
    // 6. Response 생성
    // ------------------------------------------------------------
    std::vector<Rt::Response> result;
    result.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        bool done = output_tokens[i] == C::GPT2_EOS_TOKEN_ID;

        result.emplace_back(
            reqs[i]->request_id,
            output_tokens[i],
            done
        );
    }

    return result;
}

}