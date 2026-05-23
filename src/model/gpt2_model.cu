#include "model/gpt2_model.cuh"
#include "model/gpt2_common.cuh"   // layernorm, gelu, linear<float>, ...
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call) \
    do { cudaError_t e=(call); if(e!=cudaSuccess){ \
        fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(1);} } while(0)
#define CUBLAS_CHECK(call) \
    do { cublasStatus_t e=(call); if(e!=CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"cuBLAS error %s:%d\n",__FILE__,__LINE__); exit(1);} } while(0)

#define DECODE_BLOCK 64

/* ============================================================
 * Paged Decode MHA 커널
 *
 * Pool layout (block p, slot s, head h):
 *   K: pool + p*(2*B*D) + s*D + h*d_head
 *   V: pool + p*(2*B*D) + B*D + s*D + h*d_head
 *   B = block_size, D = d_model
 * ============================================================ */
__global__ void paged_decode_mha_kernel(
    const float* __restrict__ Q,           // [d_model]
    const int*   __restrict__ block_table, // [num_logical_blocks]
    const float* __restrict__ kv_pool,
    float*       __restrict__ O,           // [d_model]
    int num_tokens, int block_size,
    int d_model, int d_head, float scale)
{
    extern __shared__ float S[];
    __shared__ float rbuf[DECODE_BLOCK];
    __shared__ float s_max, s_sum;

    int h   = blockIdx.x;
    int tid = threadIdx.x;
    int bs  = blockDim.x;
    const float* Qh = Q + h * d_head;
    size_t BH = (size_t)block_size * d_model;

    /* (1) Attention scores */
    for (int j = tid; j < num_tokens; j += bs) {
        int lb = j / block_size, slot = j % block_size;
        int phys = block_table[lb];
        const float* Kj = kv_pool + phys*2*BH + slot*d_model + h*d_head;
        float acc = 0.f;
        for (int k = 0; k < d_head; k++) acc += Qh[k] * Kj[k];
        S[j] = acc * scale;
    }
    __syncthreads();

    /* (2) Row max */
    float lm = -INFINITY;
    for (int j = tid; j < num_tokens; j += bs) lm = fmaxf(lm, S[j]);
    rbuf[tid] = lm; __syncthreads();
    for (int s = bs/2; s > 0; s >>= 1) {
        if (tid < s) rbuf[tid] = fmaxf(rbuf[tid], rbuf[tid+s]); __syncthreads();
    }
    if (tid == 0) s_max = rbuf[0]; __syncthreads();

    /* (3) Exp & sum */
    float ls = 0.f;
    for (int j = tid; j < num_tokens; j += bs) {
        float e = expf(S[j] - s_max); S[j] = e; ls += e;
    }
    rbuf[tid] = ls; __syncthreads();
    for (int s = bs/2; s > 0; s >>= 1) {
        if (tid < s) rbuf[tid] += rbuf[tid+s]; __syncthreads();
    }
    if (tid == 0) s_sum = rbuf[0]; __syncthreads();
    float inv = 1.f / s_sum;

    /* (4) O[h*d_head + k] = sum_j S[j] * V[j][h][k] */
    for (int k = tid; k < d_head; k += bs) {
        float acc = 0.f;
        for (int j = 0; j < num_tokens; j++) {
            int lb = j / block_size, slot = j % block_size;
            int phys = block_table[lb];
            const float* Vj = kv_pool + phys*2*BH + BH + slot*d_model + h*d_head;
            acc += S[j] * Vj[k];
        }
        O[h * d_head + k] = acc * inv;
    }
}

/* ============================================================
 * Prefill MHA
 * ============================================================ */
static void mha_prefill(cublasHandle_t handle,
                        const float* x,
                        const float* qkv_w, const float* qkv_b,
                        const float* out_w, const float* out_b,
                        float* attn_out,
                        float* buf_qkv, float* buf_S, float* buf_O,
                        PagedKVCache& kv, int seq_len)
{
    const float alpha = 1.f, beta = 0.f, scale = 1.f / sqrtf((float)D_HEAD);

    // 1) QKV projection
    linear<float>(handle, x, qkv_w, qkv_b, buf_qkv, seq_len, D_MODEL, 3*D_MODEL);

    // 2) K, V pack → PagedKVCache에 저장
    {
        float *d_k, *d_v;
        CUDA_CHECK(cudaMalloc(&d_k, (size_t)seq_len * D_MODEL * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, (size_t)seq_len * D_MODEL * sizeof(float)));

        // buf_qkv: [seq, 3*D] interleaved → K at offset D_MODEL, V at 2*D_MODEL
        float* h_qkv = (float*)malloc((size_t)seq_len * 3 * D_MODEL * sizeof(float));
        float* h_k   = (float*)malloc((size_t)seq_len * D_MODEL * sizeof(float));
        float* h_v   = (float*)malloc((size_t)seq_len * D_MODEL * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_qkv, buf_qkv,
            (size_t)seq_len * 3 * D_MODEL * sizeof(float),
            cudaMemcpyDeviceToHost));
        for (int s = 0; s < seq_len; s++) {
            memcpy(h_k + s*D_MODEL, h_qkv + s*3*D_MODEL + D_MODEL,   D_MODEL*sizeof(float));
            memcpy(h_v + s*D_MODEL, h_qkv + s*3*D_MODEL + 2*D_MODEL, D_MODEL*sizeof(float));
        }
        CUDA_CHECK(cudaMemcpy(d_k, h_k, (size_t)seq_len*D_MODEL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v, (size_t)seq_len*D_MODEL*sizeof(float), cudaMemcpyHostToDevice));
        kv.append_token_kv_batch(d_k, d_v, seq_len);
        cudaFree(d_k); cudaFree(d_v);
        free(h_qkv); free(h_k); free(h_v);
    }

    // 3) QK^T — strided batched (interleaved Q, K)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, D_HEAD,
        &scale,
        buf_qkv + D_MODEL,  3*D_MODEL, D_HEAD,   // K
        buf_qkv,            3*D_MODEL, D_HEAD,   // Q
        &beta,
        buf_S, seq_len, (long long)seq_len*seq_len,
        N_HEADS));

    // 4) Causal mask + Softmax
    causal_mask_apply(buf_S, seq_len, N_HEADS);
    softmax_prefill(buf_S, seq_len, N_HEADS);

    // 5) O = softmax(S) * V
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D_HEAD, seq_len, seq_len,
        &alpha,
        buf_qkv + 2*D_MODEL, 3*D_MODEL, D_HEAD,
        buf_S,               seq_len,   (long long)seq_len*seq_len,
        &beta,
        buf_O, D_MODEL, D_HEAD,
        N_HEADS));

    // 6) Output projection
    linear<float>(handle, buf_O, out_w, out_b, attn_out, seq_len, D_MODEL, D_MODEL);
}

/* ============================================================
 * Decode MHA — Paged Attention
 * ============================================================ */
static void mha_decode(cublasHandle_t handle,
                       const float* x_in,
                       const float* qkv_w, const float* qkv_b,
                       const float* out_w, const float* out_b,
                       float* attn_out,
                       float* buf_qkv, float* buf_O,
                       PagedKVCache& kv, int* d_block_table)
{
    // 1) QKV projection (seq=1)
    linear<float>(handle, x_in, qkv_w, qkv_b, buf_qkv, 1, D_MODEL, 3*D_MODEL);

    // 2) 현재 토큰의 K, V를 KV cache에 추가
    kv.append_token_kv(buf_qkv + D_MODEL, buf_qkv + 2*D_MODEL);

    // 3) block_table → GPU 복사
    const auto& bt = kv.get_block_table();
    int num_blocks = (int)bt.size();
    int num_tokens = kv.get_num_tokens();
    std::vector<int> h_phys(num_blocks);
    for (int i = 0; i < num_blocks; i++) h_phys[i] = bt[i].phys_block_id;
    CUDA_CHECK(cudaMemcpy(d_block_table, h_phys.data(),
                          num_blocks * sizeof(int), cudaMemcpyHostToDevice));

    // 4) Paged decode attention
    float scale = 1.f / sqrtf((float)D_HEAD);
    float* pool = BlockAllocator::getInstance().get_pool();
    int    bsz  = BlockAllocator::getInstance().get_block_size();
    size_t smem = (size_t)num_tokens * sizeof(float);

    paged_decode_mha_kernel<<<N_HEADS, DECODE_BLOCK, smem>>>(
        buf_qkv, d_block_table, pool, buf_O,
        num_tokens, bsz, D_MODEL, D_HEAD, scale);

    // 5) Output projection
    linear<float>(handle, buf_O, out_w, out_b, attn_out, 1, D_MODEL, D_MODEL);
}

/* ============================================================
 * Transformer Block
 * ============================================================ */
static void block_prefill(cublasHandle_t handle, float* x,
                          const GPT2Weights& W, int layer,
                          float* buf_ln, float* buf_qkv,
                          float* buf_S, float* buf_O,
                          float* buf_attn, float* buf_ff,
                          PagedKVCache& kv, int seq_len)
{
    int nd = seq_len * D_MODEL;
    layernorm(x, buf_ln, W.ln1_w[layer], W.ln1_b[layer], seq_len, D_MODEL);
    mha_prefill(handle, buf_ln,
                W.qkv_w[layer], W.qkv_b[layer],
                W.out_w[layer], W.out_b[layer],
                buf_attn, buf_qkv, buf_S, buf_O, kv, seq_len);
    residual_add(x, buf_attn, nd);

    layernorm(x, buf_ln, W.ln2_w[layer], W.ln2_b[layer], seq_len, D_MODEL);
    linear<float>(handle, buf_ln, W.fc1_w[layer], W.fc1_b[layer], buf_ff, seq_len, D_MODEL, D_FF);
    gelu(buf_ff, seq_len * D_FF);
    linear<float>(handle, buf_ff, W.fc2_w[layer], W.fc2_b[layer], buf_attn, seq_len, D_FF, D_MODEL);
    residual_add(x, buf_attn, nd);
}

static void block_decode(cublasHandle_t handle, float* x_row,
                         const GPT2Weights& W, int layer,
                         float* buf_ln, float* buf_qkv, float* buf_O,
                         float* buf_attn, float* buf_ff,
                         PagedKVCache& kv, int* d_block_table)
{
    layernorm(x_row, buf_ln, W.ln1_w[layer], W.ln1_b[layer], 1, D_MODEL);
    mha_decode(handle, buf_ln,
               W.qkv_w[layer], W.qkv_b[layer],
               W.out_w[layer], W.out_b[layer],
               buf_attn, buf_qkv, buf_O, kv, d_block_table);
    residual_add(x_row, buf_attn, D_MODEL);

    layernorm(x_row, buf_ln, W.ln2_w[layer], W.ln2_b[layer], 1, D_MODEL);
    linear<float>(handle, buf_ln, W.fc1_w[layer], W.fc1_b[layer], buf_ff, 1, D_MODEL, D_FF);
    gelu(buf_ff, D_FF);
    linear<float>(handle, buf_ff, W.fc2_w[layer], W.fc2_b[layer], buf_attn, 1, D_FF, D_MODEL);
    residual_add(x_row, buf_attn, D_MODEL);
}

/* ============================================================
 * Weight 로더
 * ============================================================ */
static float* alloc_gpu(size_t n) {
    float* p; CUDA_CHECK(cudaMalloc(&p, n*sizeof(float))); return p;
}

static float* load_bin(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* h = (float*)malloc(n * sizeof(float));
    fread(h, sizeof(float), n, f); fclose(f);
    float* d = alloc_gpu(n);
    CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
    free(h); return d;
}

static GPT2Weights load_weights(const char* dir) {
    GPT2Weights W; char path[512];
#define LOAD(ptr, fname, n) \
    snprintf(path, 512, "%s/%s", dir, fname); ptr = load_bin(path, (size_t)(n));
    LOAD(W.wte, "wte.bin", (size_t)VOCAB*D_MODEL);
    LOAD(W.wpe, "wpe.bin", (size_t)MAX_SEQ*D_MODEL);
    for (int l = 0; l < N_LAYERS; l++) {
        char nm[64];
        snprintf(nm,64,"ln1_w_%d.bin",l); LOAD(W.ln1_w[l], nm, D_MODEL);
        snprintf(nm,64,"ln1_b_%d.bin",l); LOAD(W.ln1_b[l], nm, D_MODEL);
        snprintf(nm,64,"qkv_w_%d.bin",l); LOAD(W.qkv_w[l], nm, (size_t)3*D_MODEL*D_MODEL);
        snprintf(nm,64,"qkv_b_%d.bin",l); LOAD(W.qkv_b[l], nm, 3*D_MODEL);
        snprintf(nm,64,"out_w_%d.bin",l); LOAD(W.out_w[l], nm, (size_t)D_MODEL*D_MODEL);
        snprintf(nm,64,"out_b_%d.bin",l); LOAD(W.out_b[l], nm, D_MODEL);
        snprintf(nm,64,"ln2_w_%d.bin",l); LOAD(W.ln2_w[l], nm, D_MODEL);
        snprintf(nm,64,"ln2_b_%d.bin",l); LOAD(W.ln2_b[l], nm, D_MODEL);
        snprintf(nm,64,"fc1_w_%d.bin",l); LOAD(W.fc1_w[l], nm, (size_t)D_FF*D_MODEL);
        snprintf(nm,64,"fc1_b_%d.bin",l); LOAD(W.fc1_b[l], nm, D_FF);
        snprintf(nm,64,"fc2_w_%d.bin",l); LOAD(W.fc2_w[l], nm, (size_t)D_MODEL*D_FF);
        snprintf(nm,64,"fc2_b_%d.bin",l); LOAD(W.fc2_b[l], nm, D_MODEL);
    }
    LOAD(W.ln_f_w, "ln_f_w.bin", D_MODEL);
    LOAD(W.ln_f_b, "ln_f_b.bin", D_MODEL);
#undef LOAD
    return W;
}

static int argmax_cpu(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* ============================================================
 * GPT2Model 구현
 * ============================================================ */
static GPT2Model* g_instance = nullptr;

GPT2Model::GPT2Model(const char* weight_dir, int blk_size)
    : block_size(blk_size)
{
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("Loading GPT-2 weights (FP32)... "); fflush(stdout);
    W = load_weights(weight_dir);
    printf("done\n");

    buf_x      = alloc_gpu((size_t)MAX_SEQ * D_MODEL);
    buf_ln     = alloc_gpu((size_t)MAX_SEQ * D_MODEL);
    buf_qkv    = alloc_gpu((size_t)MAX_SEQ * 3 * D_MODEL);
    buf_S      = alloc_gpu((size_t)N_HEADS * MAX_SEQ * MAX_SEQ);
    buf_O      = alloc_gpu((size_t)MAX_SEQ * D_MODEL);
    buf_attn   = alloc_gpu((size_t)MAX_SEQ * D_MODEL);
    buf_ff     = alloc_gpu((size_t)MAX_SEQ * D_FF);
    buf_logits = alloc_gpu((size_t)VOCAB);

    int max_blocks = MAX_SEQ / blk_size + 1;
    CUDA_CHECK(cudaMalloc(&d_block_table, max_blocks * sizeof(int)));
}

void GPT2Model::init(const char* weight_dir, int blk_size) {
    if (!g_instance)
        g_instance = new GPT2Model(weight_dir, blk_size);
}

GPT2Model& GPT2Model::get() {
    if (!g_instance) {
        fprintf(stderr, "GPT2Model::get() called before init()\n");
        exit(1);
    }
    return *g_instance;
}

int GPT2Model::prefill(const int* d_token_ids, int prompt_len, PagedKVCache& kv) {
    // Embedding
    embedding_lookup(d_token_ids, W.wte, W.wpe, buf_x, prompt_len, D_MODEL, 0);

    // Transformer layers
    for (int l = 0; l < N_LAYERS; l++)
        block_prefill(handle, buf_x, W, l,
                      buf_ln, buf_qkv, buf_S, buf_O, buf_attn, buf_ff,
                      kv, prompt_len);

    // Final LN + LM head (마지막 토큰 위치)
    float* last_x = buf_x + (size_t)(prompt_len - 1) * D_MODEL;
    layernorm(last_x, buf_ln, W.ln_f_w, W.ln_f_b, 1, D_MODEL);
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        VOCAB, 1, D_MODEL,
        &alpha, W.wte, D_MODEL, buf_ln, D_MODEL,
        &beta,  buf_logits, VOCAB));

    CUDA_CHECK(cudaDeviceSynchronize());
    float* h_logits = (float*)malloc(VOCAB * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, buf_logits, VOCAB*sizeof(float), cudaMemcpyDeviceToHost));
    int next = argmax_cpu(h_logits, VOCAB);
    free(h_logits);
    return next;
}

int GPT2Model::decode_step(int token_id, PagedKVCache& kv) {
    int pos = kv.get_num_tokens();
    float* x_row = buf_x + (size_t)pos * D_MODEL;

    // Embedding (position = pos)
    int* d_tok; CUDA_CHECK(cudaMalloc(&d_tok, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tok, &token_id, sizeof(int), cudaMemcpyHostToDevice));
    embedding_lookup(d_tok, W.wte, W.wpe, x_row, 1, D_MODEL, pos);
    cudaFree(d_tok);

    // Transformer layers
    for (int l = 0; l < N_LAYERS; l++)
        block_decode(handle, x_row, W, l,
                     buf_ln, buf_qkv, buf_O, buf_attn, buf_ff,
                     kv, d_block_table);

    // Final LN + LM head
    layernorm(x_row, buf_ln, W.ln_f_w, W.ln_f_b, 1, D_MODEL);
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        VOCAB, 1, D_MODEL,
        &alpha, W.wte, D_MODEL, buf_ln, D_MODEL,
        &beta,  buf_logits, VOCAB));

    CUDA_CHECK(cudaDeviceSynchronize());
    float* h_logits = (float*)malloc(VOCAB * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, buf_logits, VOCAB*sizeof(float), cudaMemcpyDeviceToHost));
    int next = argmax_cpu(h_logits, VOCAB);
    free(h_logits);
    return next;
}
