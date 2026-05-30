#include "model/gpt2_wmma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <nvtx3/nvToolsExt.h>
#include "kernel/flashattention1.cuh"

#ifdef USE_FLASH_ATTENTION_PREFILL
#define USE_FLASH_ATTENTION_PREFILL 0
#endif

#define CUDA_CHECK(call) \
    do { cudaError_t e=(call); if(e!=cudaSuccess){ \
        fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(1);} } while(0)
#define CUBLAS_CHECK(call) \
    do { cublasStatus_t e=(call); if(e!=CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"cuBLAS error %s:%d\n",__FILE__,__LINE__); exit(1);} } while(0)

#define DECODE_BLOCK 64

/* ============================================================
 * linear_wmma
 *
 * y = x_fp32 @ W_T_fp16 + b
 *   x    : [seq_len, in_feat]  FP32
 *   W_T  : [in_feat, out_feat] FP16  (전치 저장)
 *   y    : [seq_len, out_feat] FP32
 *   buf_xh: FP16 스크래치 [seq_len, in_feat]
 *
 * wmma_gemm(A, B, C, M, N, K) = C = A @ B
 *   A = x_fp16  [seq_len, in_feat]
 *   B = W_T     [in_feat, out_feat]
 *   → C = y     [seq_len, out_feat]   ✓
 * ============================================================ */
static void linear_wmma(const float* x, const __half* W_T, const float* b, float* y,
                         int seq_len, int in_feat, int out_feat,
                         __half* buf_xh)
{
    // FP32 → FP16 변환 (buf_xh에 저장)
    float_to_half_device(x, buf_xh, seq_len * in_feat);

    // WMMA GEMM: y = x_fp16 @ W_T_fp16
    wmma_gemm(buf_xh, W_T, y, seq_len, out_feat, in_feat);

    // Bias
    if (b) bias_add(y, b, seq_len, out_feat);
}

/* ============================================================
 * Paged Decode MHA 커널 (gpt2_model.cu와 동일 — FP32 어텐션)
 * ============================================================ */
__global__ void paged_decode_mha_wmma_kernel(
    const float* __restrict__ Q,
    const int*   __restrict__ block_table,
    const float* __restrict__ kv_pool,
    float*       __restrict__ O,
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

    /* (4) O = softmax(S) @ V */
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
 *
 * QKV / Out projection → WMMA
 * QK^T, SV             → cuBLAS strided batched (FP32)
 * ============================================================ */
static void mha_prefill(cublasHandle_t handle,
                        const float* x,
                        const __half* qkv_w, const float* qkv_b,
                        const __half* out_w, const float* out_b,
                        float* attn_out,
                        float* buf_qkv, float* buf_S, float* buf_O,
                        __half* buf_xh,
                        PagedKVCache& kv, int seq_len)
{
    const float alpha = 1.f, beta = 0.f;
    const float scale = 1.f / sqrtf((float)D_HEAD);

    // 1) QKV projection — WMMA
    //    x [seq, D] @ qkv_w [D, 3D] → buf_qkv [seq, 3D]
    nvtxRangePushA("qkv_proj");
    linear_wmma(x, qkv_w, qkv_b, buf_qkv, seq_len, D_MODEL, 3*D_MODEL, buf_xh);
    nvtxRangePop();
    // 2) K, V → PagedKVCache 저장
    {
	nvtxRangePushA("kv_append");
        /*
	float *d_k, *d_v;
        float* h_qkv = (float*)malloc((size_t)seq_len * 3*D_MODEL * sizeof(float));
        float* h_k   = (float*)malloc((size_t)seq_len * D_MODEL   * sizeof(float));
        float* h_v   = (float*)malloc((size_t)seq_len * D_MODEL   * sizeof(float));

        CUDA_CHECK(cudaMemcpy(h_qkv, buf_qkv,
            (size_t)seq_len * 3*D_MODEL * sizeof(float),
            cudaMemcpyDeviceToHost));
        for (int s = 0; s < seq_len; s++) {
            memcpy(h_k + s*D_MODEL, h_qkv + s*3*D_MODEL +   D_MODEL, D_MODEL*sizeof(float));
            memcpy(h_v + s*D_MODEL, h_qkv + s*3*D_MODEL + 2*D_MODEL, D_MODEL*sizeof(float));
        }
        CUDA_CHECK(cudaMalloc(&d_k, (size_t)seq_len*D_MODEL*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, (size_t)seq_len*D_MODEL*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_k, h_k, (size_t)seq_len*D_MODEL*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v, (size_t)seq_len*D_MODEL*sizeof(float), cudaMemcpyHostToDevice));
        kv.append_token_kv_batch(d_k, d_v, seq_len);
        cudaFree(d_k); cudaFree(d_v);
        free(h_qkv); free(h_k); free(h_v);
	*/    
        kv.append_qkv_from_interleaved_no_alloc(buf_qkv, seq_len);
        nvtxRangePop();
    }

    // 3) QK^T — cuBLAS strided batched (interleaved Q, K in buf_qkv)
#if USE_FLASH_ATTENTION_PREFILL
    nvtxRangePushA("flashattn1_prefill");
    flashattention1_prefill(
   	buf_qkv,
        buf_O,
        seq_len,
        D_MODEL,
        D_HEAD,
        N_HEADS,
        scale
    );
    nvtxRangePop();  
#else  
    nvtxRangePushA("qk_gemm");
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    seq_len, seq_len, D_HEAD,
    &scale,
    buf_qkv + D_MODEL,  3*D_MODEL, D_HEAD,
    buf_qkv,            3*D_MODEL, D_HEAD,
    &beta,
    buf_S, seq_len, (long long)seq_len*seq_len,
    N_HEADS));
    nvtxRangePop();
    // 4) Causal mask + Softmax
    nvtxRangePushA("causal_mask");
    causal_mask_apply(buf_S, seq_len, N_HEADS);
    nvtxRangePop();

    nvtxRangePushA("softmax");
    softmax_prefill(buf_S, seq_len, N_HEADS);
    nvtxRangePop();
    // 5) O = softmax(S) * V — cuBLAS strided batched
    nvtxRangePushA("sv_gemm");
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D_HEAD, seq_len, seq_len,
        &alpha,
        buf_qkv + 2*D_MODEL, 3*D_MODEL, D_HEAD,
        buf_S,               seq_len,   (long long)seq_len*seq_len,
        &beta,
        buf_O, D_MODEL, D_HEAD,
        N_HEADS));
    nvtxRangePop();
#endif
    // 6) Output projection — WMMA
    //    buf_O [seq, D] @ out_w [D, D] → attn_out [seq, D]
    nvtxRangePushA("out_proj");
    linear_wmma(buf_O, out_w, out_b, attn_out, seq_len, D_MODEL, D_MODEL, buf_xh);
    nvtxRangePop();
}

/* ============================================================
 * Decode MHA — Paged Attention
 *
 * QKV / Out projection → WMMA
 * 어텐션 스코어 계산   → paged_decode_mha_wmma_kernel (FP32)
 * ============================================================ */
static void mha_decode(cublasHandle_t /* unused */,
                       const float* x_in,
                       const __half* qkv_w, const float* qkv_b,
                       const __half* out_w, const float* out_b,
                       float* attn_out,
                       float* buf_qkv, float* buf_O,
                       __half* buf_xh,
                       PagedKVCache& kv, int* d_block_table)
{
    // 1) QKV projection (seq=1) — WMMA
    linear_wmma(x_in, qkv_w, qkv_b, buf_qkv, 1, D_MODEL, 3*D_MODEL, buf_xh);

    // 2) 현재 토큰 K, V → KV cache 추가
    kv.append_token_kv(buf_qkv + D_MODEL, buf_qkv + 2*D_MODEL);

    // 3) block_table → GPU 복사
    const auto& bt = kv.get_block_table();
    int num_blocks = (int)bt.size();
    int num_tokens = kv.get_num_tokens();
    std::vector<int> h_phys(num_blocks);
    for (int i = 0; i < num_blocks; i++) h_phys[i] = bt[i].phys_block_id;
    CUDA_CHECK(cudaMemcpy(d_block_table, h_phys.data(),
                          num_blocks * sizeof(int), cudaMemcpyHostToDevice));

    // 4) Paged decode attention (FP32)
    float scale = 1.f / sqrtf((float)D_HEAD);
    float* pool = BlockAllocator::getInstance().get_pool();
    int    bsz  = BlockAllocator::getInstance().get_block_size();
    size_t smem = (size_t)num_tokens * sizeof(float);

    paged_decode_mha_wmma_kernel<<<N_HEADS, DECODE_BLOCK, smem>>>(
        buf_qkv, d_block_table, pool, buf_O,
        num_tokens, bsz, D_MODEL, D_HEAD, scale);

    // 5) Output projection — WMMA
    linear_wmma(buf_O, out_w, out_b, attn_out, 1, D_MODEL, D_MODEL, buf_xh);
}

/* ============================================================
 * Transformer Block
 * ============================================================ */
static void block_prefill_wmma(cublasHandle_t handle, float* x,
                                const GPT2WeightsWMMA& W, int layer,
                                float* buf_ln, float* buf_qkv,
                                float* buf_S, float* buf_O,
                                float* buf_attn, float* buf_ff,
                                __half* buf_xh,
                                PagedKVCache& kv, int seq_len)
{
    int nd = seq_len * D_MODEL;

    // Self-attention
    //nvtxRangePushA("ln1");
    layernorm(x, buf_ln, W.ln1_w[layer], W.ln1_b[layer], seq_len, D_MODEL);
    //nvtxRangePop();

    //nvtxRangePushA("mha_prefill");
    mha_prefill(handle, buf_ln,
                W.qkv_w[layer], W.qkv_b[layer],
                W.out_w[layer], W.out_b[layer],
                buf_attn, buf_qkv, buf_S, buf_O, buf_xh,
                kv, seq_len);
    //nvtxRangePop();

    residual_add(x, buf_attn, nd);

    // FFN
    //nvtxRangePushA("ln2");
    layernorm(x, buf_ln, W.ln2_w[layer], W.ln2_b[layer], seq_len, D_MODEL);
    //nvtxRangePop();
    //nvtxRangePushA("fc1");
    linear_wmma(buf_ln, W.fc1_w[layer], W.fc1_b[layer], buf_ff,
                seq_len, D_MODEL, D_FF, buf_xh);
    //nvtxRangePop();
    //nvtxRangePushA("gelu");
    gelu(buf_ff, seq_len * D_FF);
    //nvtxRangePop();
    //nvtxRangePushA("fc2");
    linear_wmma(buf_ff, W.fc2_w[layer], W.fc2_b[layer], buf_attn,
                seq_len, D_FF, D_MODEL, buf_xh);
    //nvtxRangePop();
    residual_add(x, buf_attn, nd);
}

static void block_decode_wmma(cublasHandle_t handle, float* x_row,
                               const GPT2WeightsWMMA& W, int layer,
                               float* buf_ln, float* buf_qkv, float* buf_O,
                               float* buf_attn, float* buf_ff,
                               __half* buf_xh,
                               PagedKVCache& kv, int* d_block_table)
{
    // Self-attention
    layernorm(x_row, buf_ln, W.ln1_w[layer], W.ln1_b[layer], 1, D_MODEL);
    mha_decode(handle, buf_ln,
               W.qkv_w[layer], W.qkv_b[layer],
               W.out_w[layer], W.out_b[layer],
               buf_attn, buf_qkv, buf_O, buf_xh,
               kv, d_block_table);
    residual_add(x_row, buf_attn, D_MODEL);

    // FFN
    layernorm(x_row, buf_ln, W.ln2_w[layer], W.ln2_b[layer], 1, D_MODEL);
    linear_wmma(buf_ln, W.fc1_w[layer], W.fc1_b[layer], buf_ff,
                1, D_MODEL, D_FF, buf_xh);
    gelu(buf_ff, D_FF);
    linear_wmma(buf_ff, W.fc2_w[layer], W.fc2_b[layer], buf_attn,
                1, D_FF, D_MODEL, buf_xh);
    residual_add(x_row, buf_attn, D_MODEL);
}

/* ============================================================
 * Weight 로더
 *
 * load_fp32   : FP32 binary 로드 → GPU FP32
 * load_fp16_T : FP32 binary 로드 → CPU에서 전치 → FP16 변환 → GPU
 * ============================================================ */
static float* alloc_fp32(size_t n) {
    float* p; CUDA_CHECK(cudaMalloc(&p, n*sizeof(float))); return p;
}

static float* load_fp32(const char* path, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* h = (float*)malloc(n * sizeof(float));
    fread(h, sizeof(float), n, f); fclose(f);
    float* d = alloc_fp32(n);
    CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice));
    free(h); return d;
}

/* W_orig: [rows, cols] FP32  →  W_T: [cols, rows] FP16 (GPU) */
static __half* load_fp16_T(const char* path, int rows, int cols) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    size_t n = (size_t)rows * cols;
    float*  h_f32   = (float*)  malloc(n * sizeof(float));
    float*  h_f32_T = (float*)  malloc(n * sizeof(float));
    __half* h_f16_T = (__half*) malloc(n * sizeof(__half));

    fread(h_f32, sizeof(float), n, f); fclose(f);

    // 전치: [rows, cols] → [cols, rows]
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            h_f32_T[c * rows + r] = h_f32[r * cols + c];

    // FP32 → FP16
    for (size_t i = 0; i < n; i++)
        h_f16_T[i] = __float2half(h_f32_T[i]);

    __half* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d, h_f16_T, n * sizeof(__half), cudaMemcpyHostToDevice));

    free(h_f32); free(h_f32_T); free(h_f16_T);
    return d;
}

static GPT2WeightsWMMA load_weights(const char* dir) {
    GPT2WeightsWMMA W;
    char path[512];

#define LOADF(ptr, fname, n) \
    snprintf(path, 512, "%s/%s", dir, fname); ptr = load_fp32(path, (size_t)(n));
#define LOADT(ptr, fname, rows, cols) \
    snprintf(path, 512, "%s/%s", dir, fname); ptr = load_fp16_T(path, rows, cols);

    // 임베딩 (FP32)
    LOADF(W.wte, "wte.bin", (size_t)VOCAB * D_MODEL);
    LOADF(W.wpe, "wpe.bin", (size_t)MAX_SEQ * D_MODEL);

    for (int l = 0; l < N_LAYERS; l++) {
        char nm[64];

        // LN (FP32)
        snprintf(nm,64,"ln1_w_%d.bin",l); LOADF(W.ln1_w[l], nm, D_MODEL);
        snprintf(nm,64,"ln1_b_%d.bin",l); LOADF(W.ln1_b[l], nm, D_MODEL);
        snprintf(nm,64,"ln2_w_%d.bin",l); LOADF(W.ln2_w[l], nm, D_MODEL);
        snprintf(nm,64,"ln2_b_%d.bin",l); LOADF(W.ln2_b[l], nm, D_MODEL);

        // Bias (FP32)
        snprintf(nm,64,"qkv_b_%d.bin",l); LOADF(W.qkv_b[l], nm, 3*D_MODEL);
        snprintf(nm,64,"out_b_%d.bin",l); LOADF(W.out_b[l], nm, D_MODEL);
        snprintf(nm,64,"fc1_b_%d.bin",l); LOADF(W.fc1_b[l], nm, D_FF);
        snprintf(nm,64,"fc2_b_%d.bin",l); LOADF(W.fc2_b[l], nm, D_MODEL);

        // 가중치 — FP16 전치 저장
        //   파일 shape     저장 shape (전치)
        snprintf(nm,64,"qkv_w_%d.bin",l); LOADT(W.qkv_w[l], nm, 3*D_MODEL, D_MODEL); // [3D,D]→[D,3D]
        snprintf(nm,64,"out_w_%d.bin",l); LOADT(W.out_w[l], nm,   D_MODEL, D_MODEL); // [D,D]→[D,D]
        snprintf(nm,64,"fc1_w_%d.bin",l); LOADT(W.fc1_w[l], nm,     D_FF,  D_MODEL); // [FF,D]→[D,FF]
        snprintf(nm,64,"fc2_w_%d.bin",l); LOADT(W.fc2_w[l], nm,   D_MODEL,   D_FF ); // [D,FF]→[FF,D]
    }
    LOADF(W.ln_f_w, "ln_f_w.bin", D_MODEL);
    LOADF(W.ln_f_b, "ln_f_b.bin", D_MODEL);

#undef LOADF
#undef LOADT
    return W;
}

static int argmax_cpu(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* ============================================================
 * GPT2ModelWMMA 구현
 * ============================================================ */
static GPT2ModelWMMA* g_wmma_instance = nullptr;

GPT2ModelWMMA::GPT2ModelWMMA(const char* weight_dir, int blk_size)
    : block_size(blk_size)
{
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("Loading GPT-2 weights (WMMA FP16)... "); fflush(stdout);
    W = load_weights(weight_dir);
    printf("done\n");

    // FP32 스크래치
    auto af = [](size_t n) -> float* {
        float* p; cudaMalloc(&p, n*sizeof(float)); return p;
    };
    buf_x      = af((size_t)MAX_SEQ * D_MODEL);
    buf_ln     = af((size_t)MAX_SEQ * D_MODEL);
    buf_qkv    = af((size_t)MAX_SEQ * 3*D_MODEL);
    buf_S      = af((size_t)N_HEADS * MAX_SEQ * MAX_SEQ);
    buf_O      = af((size_t)MAX_SEQ * D_MODEL);
    buf_attn   = af((size_t)MAX_SEQ * D_MODEL);
    buf_ff     = af((size_t)MAX_SEQ * D_FF);
    buf_logits = af((size_t)VOCAB);

    // FP16 스크래치 — FC2 입력 [MAX_SEQ, D_FF] 가 최대 크기
    CUDA_CHECK(cudaMalloc(&buf_xh, (size_t)MAX_SEQ * D_FF * sizeof(__half)));

    int max_blocks = MAX_SEQ / blk_size + 1;
    CUDA_CHECK(cudaMalloc(&d_block_table, max_blocks * sizeof(int)));

    // batch_decode 전용 버퍼
    CUDA_CHECK(cudaMalloc(&d_batch_block_tables,
        (size_t)MAX_BATCH * max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens, MAX_BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf_batch_logits, (size_t)MAX_BATCH * VOCAB * sizeof(float)));
}

void GPT2ModelWMMA::init(const char* weight_dir, int blk_size) {
    if (!g_wmma_instance)
        g_wmma_instance = new GPT2ModelWMMA(weight_dir, blk_size);
}

GPT2ModelWMMA& GPT2ModelWMMA::get() {
    if (!g_wmma_instance) {
        fprintf(stderr, "GPT2ModelWMMA::get() called before init()\n");
        exit(1);
    }
    return *g_wmma_instance;
}

int GPT2ModelWMMA::prefill(
    const int* d_token_ids,
    int prompt_len,
    std::vector<PagedKVCache>& layer_kv
) {
    /*
     * 핵심 최적화:
     * prompt_len이 정해졌으므로 prefill에 필요한 KV block 수가 결정된다.
     * 모든 layer에 대해 미리 physical block과 d_block_table을 준비한다.
     *
     * 이후 mha_prefill() 안에서는 append_qkv_from_interleaved_no_alloc()
     * 을 사용해서 K/V write kernel만 실행한다.
     */
    nvtxRangePushA("prefill_reserve_kv");
    for (int l = 0; l < N_LAYERS; l++) {
        layer_kv[l].reserve_blocks_for_tokens(prompt_len);
    }
    nvtxRangePop();

    // Embedding
    nvtxRangePushA("prefill_embedding");
    embedding_lookup(
        d_token_ids,
        W.wte,
        W.wpe,
        buf_x,
        prompt_len,
        D_MODEL,
        0
    );
    nvtxRangePop();

    // Transformer layers
    for (int l = 0; l < N_LAYERS; l++) {
        char range_name[64];
        snprintf(range_name, sizeof(range_name), "prefill_layer_%d", l);

        nvtxRangePushA(range_name);
        block_prefill_wmma(
            handle,
            buf_x,
            W,
            l,
            buf_ln,
            buf_qkv,
            buf_S,
            buf_O,
            buf_attn,
            buf_ff,
            buf_xh,
            layer_kv[l],
            prompt_len
        );
        nvtxRangePop();
    }

    // Final LN + LM head
    float* last_x = buf_x + (size_t)(prompt_len - 1) * D_MODEL;

    layernorm(
        last_x,
        buf_ln,
        W.ln_f_w,
        W.ln_f_b,
        1,
        D_MODEL
    );

    const float alpha = 1.f;
    const float beta  = 0.f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        VOCAB,
        1,
        D_MODEL,
        &alpha,
        W.wte,
        D_MODEL,
        buf_ln,
        D_MODEL,
        &beta,
        buf_logits,
        VOCAB
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_logits = (float*)malloc(VOCAB * sizeof(float));

    CUDA_CHECK(cudaMemcpy(
        h_logits,
        buf_logits,
        VOCAB * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    int next = argmax_cpu(h_logits, VOCAB);

    free(h_logits);

    return next;
}

int GPT2ModelWMMA::decode_step(int token_id, std::vector<PagedKVCache>& layer_kv) {
    int    pos   = layer_kv[0].get_num_tokens();
    float* x_row = buf_x + (size_t)pos * D_MODEL;

    // Embedding
    int* d_tok; CUDA_CHECK(cudaMalloc(&d_tok, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tok, &token_id, sizeof(int), cudaMemcpyHostToDevice));
    embedding_lookup(d_tok, W.wte, W.wpe, x_row, 1, D_MODEL, pos);
    cudaFree(d_tok);

    // Transformer layers
    for (int l = 0; l < N_LAYERS; l++)
        block_decode_wmma(handle, x_row, W, l,
                          buf_ln, buf_qkv, buf_O,
                          buf_attn, buf_ff, buf_xh,
                          layer_kv[l], d_block_table);

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

/* ============================================================
 * Batched Paged Decode MHA 커널
 *
 * grid: dim3(N_HEADS, B)
 *   blockIdx.x = head index
 *   blockIdx.y = batch index (시퀀스 번호)
 *
 * block_tables: [B, max_blocks] flattened
 * seq_lens    : [B] 각 시퀀스의 현재 토큰 수
 * ============================================================ */
__global__ void paged_decode_mha_batch_kernel(
    const float* __restrict__ Q,             // [B, D_MODEL]
    const int*   __restrict__ block_tables,  // [B, max_blocks]
    const int*   __restrict__ seq_lens,      // [B]
    const float* __restrict__ kv_pool,
    float*       __restrict__ O,             // [B, D_MODEL]
    int B, int max_blocks,
    int block_size, int d_model, int d_head,
    float scale)
{
    extern __shared__ float S[];   // [seq_lens[b]] — dynamic

    int h   = blockIdx.x;   // head
    int b   = blockIdx.y;   // batch
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    int num_tokens = seq_lens[b];
    const float* Qh         = Q            + b * d_model + h * d_head;
    const int*   block_table = block_tables + b * max_blocks;
    float*       Oh          = O            + b * d_model + h * d_head;
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

    /* (2) Softmax */
    __shared__ float s_max, s_sum;
    __shared__ float rbuf[DECODE_BLOCK];

    float lm = -INFINITY;
    for (int j = tid; j < num_tokens; j += bs) lm = fmaxf(lm, S[j]);
    rbuf[tid] = lm; __syncthreads();
    for (int s = bs/2; s > 0; s >>= 1) {
        if (tid < s) rbuf[tid] = fmaxf(rbuf[tid], rbuf[tid+s]); __syncthreads();
    }
    if (tid == 0) s_max = rbuf[0]; __syncthreads();

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

    /* (3) O = softmax(S) @ V */
    for (int k = tid; k < d_head; k += bs) {
        float acc = 0.f;
        for (int j = 0; j < num_tokens; j++) {
            int lb = j / block_size, slot = j % block_size;
            int phys = block_table[lb];
            const float* Vj = kv_pool + phys*2*BH + BH + slot*d_model + h*d_head;
            acc += S[j] * Vj[k];
        }
        Oh[k] = acc * inv;
    }
}

/* ============================================================
 * batch_decode
 * ============================================================ */
std::vector<int> GPT2ModelWMMA::batch_decode(
    const std::vector<Request*>& reqs
) {
    int B = (int)reqs.size();

    if (B == 0) {
        return {};
    }

    if (B > MAX_BATCH) {
        fprintf(stderr,
                "GPT2ModelWMMA::batch_decode: B=%d exceeds MAX_BATCH=%d\n",
                B,
                MAX_BATCH);
        exit(1);
    }

    int max_blocks = MAX_SEQ / block_size + 1;

    const float scale = 1.f / sqrtf((float)D_HEAD);
    const float alpha = 1.f;
    const float beta  = 0.f;

    float* pool = BlockAllocator::getInstance().get_pool();

    std::vector<int> h_seq_lens(B, 0);
    std::vector<int> h_block_tables((size_t)B * max_blocks, 0);

    /*
     * 1. 각 request의 현재 decode input token을 embedding해서
     *    buf_x에 [B, D_MODEL] 형태로 쌓는다.
     *
     * position은 layer_kv[0] 기준으로 잡는다.
     * 정상적인 layer별 KV 구조라면 모든 layer의 num_tokens는 같아야 한다.
     */
    for (int b = 0; b < B; b++) {
        Request* req = reqs[b];

        if ((int)req->layer_kv.size() != N_LAYERS) {
            fprintf(stderr,
                    "GPT2ModelWMMA::batch_decode: req id=%d has layer_kv.size()=%zu, expected %d\n",
                    req->id,
                    req->layer_kv.size(),
                    N_LAYERS);
            exit(1);
        }

        int pos = req->layer_kv[0].get_num_tokens();

        if (pos >= MAX_SEQ) {
            fprintf(stderr,
                    "GPT2ModelWMMA::batch_decode: req id=%d position=%d exceeds MAX_SEQ=%d\n",
                    req->id,
                    pos,
                    MAX_SEQ);
            exit(1);
        }

        int token_id = req->output_ids.empty()
            ? req->prompt_ids.back()
            : req->output_ids.back();

        int* d_tok = nullptr;

        CUDA_CHECK(cudaMalloc(&d_tok, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(
            d_tok,
            &token_id,
            sizeof(int),
            cudaMemcpyHostToDevice
        ));

        embedding_lookup(
            d_tok,
            W.wte,
            W.wpe,
            buf_x + (size_t)b * D_MODEL,
            1,
            D_MODEL,
            pos
        );

        CUDA_CHECK(cudaFree(d_tok));
    }

    /*
     * 2. Transformer layers
     *
     * 핵심:
     * - layer l에서는 reqs[b]->layer_kv[l]만 사용한다.
     * - block table도 layer마다 다시 만들어야 한다.
     * - 현재 paged_decode_mha_batch_kernel은 Q를 [B, D_MODEL]로 읽으므로,
     *   buf_qkv의 Q 부분만 buf_ln에 복사해서 kernel에 넘긴다.
     */
    for (int l = 0; l < N_LAYERS; l++) {
        /*
         * 2-1. LN1
         * buf_x  : [B, D_MODEL]
         * buf_ln : [B, D_MODEL]
         */
        layernorm(
            buf_x,
            buf_ln,
            W.ln1_w[l],
            W.ln1_b[l],
            B,
            D_MODEL
        );

        /*
         * 2-2. QKV projection
         * buf_qkv: [B, 3 * D_MODEL]
         *
         * row b layout:
         *   buf_qkv[b] = [Q | K | V]
         */
        linear_wmma(
            buf_ln,
            W.qkv_w[l],
            W.qkv_b[l],
            buf_qkv,
            B,
            D_MODEL,
            3 * D_MODEL,
            buf_xh
        );

        /*
         * 2-3. 현재 layer의 K/V를 각 request의 layer_kv[l]에 append.
         *
         * append 후 attention에서 현재 token까지 포함해야 하므로
         * h_seq_lens[b]는 append 이후의 길이를 사용한다.
         */
        std::fill(
            h_seq_lens.begin(),
            h_seq_lens.end(),
            0
        );

        std::fill(
            h_block_tables.begin(),
            h_block_tables.end(),
            0
        );

        for (int b = 0; b < B; b++) {
            Request* req = reqs[b];

            float* qkv_b_ptr = buf_qkv + (size_t)b * 3 * D_MODEL;

            req->layer_kv[l].append_token_kv(
                qkv_b_ptr + D_MODEL,
                qkv_b_ptr + 2 * D_MODEL
            );

            int seq_len = req->layer_kv[l].get_num_tokens();

            if (seq_len > MAX_SEQ) {
                fprintf(stderr,
                        "GPT2ModelWMMA::batch_decode: req id=%d layer=%d seq_len=%d exceeds MAX_SEQ=%d\n",
                        req->id,
                        l,
                        seq_len,
                        MAX_SEQ);
                exit(1);
            }

            h_seq_lens[b] = seq_len;

            const auto& bt = req->layer_kv[l].get_block_table();

            if ((int)bt.size() > max_blocks) {
                fprintf(stderr,
                        "GPT2ModelWMMA::batch_decode: req id=%d layer=%d block_table.size()=%zu exceeds max_blocks=%d\n",
                        req->id,
                        l,
                        bt.size(),
                        max_blocks);
                exit(1);
            }

            for (int i = 0; i < (int)bt.size(); i++) {
                h_block_tables[(size_t)b * max_blocks + i] =
                    bt[i].phys_block_id;
            }
        }

        /*
         * 2-4. 현재 kernel은 Q를 [B, D_MODEL] contiguous로 읽는다.
         *
         * 그런데 buf_qkv는 [B, 3 * D_MODEL]라서 그대로 넘기면
         * b > 0에서 Q offset이 틀어진다.
         *
         * 따라서 Q 부분만 buf_ln에 복사해서:
         *   buf_ln[b] = Q_b
         * 로 만든 뒤 paged attention kernel에 넘긴다.
         *
         * 이 시점 이후 LN1 결과는 필요 없고,
         * buf_ln은 FFN 앞 LN2에서 다시 덮어쓰므로 재사용해도 된다.
         */
        for (int b = 0; b < B; b++) {
            CUDA_CHECK(cudaMemcpy(
                buf_ln + (size_t)b * D_MODEL,
                buf_qkv + (size_t)b * 3 * D_MODEL,
                (size_t)D_MODEL * sizeof(float),
                cudaMemcpyDeviceToDevice
            ));
        }

        /*
         * 2-5. seq_lens와 block tables를 GPU로 복사.
         * block table은 layer마다 다를 수 있으므로 매 layer마다 갱신한다.
         */
        CUDA_CHECK(cudaMemcpy(
            d_seq_lens,
            h_seq_lens.data(),
            (size_t)B * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_batch_block_tables,
            h_block_tables.data(),
            (size_t)B * max_blocks * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        /*
         * 2-6. Batched paged attention.
         *
         * grid.x = head
         * grid.y = batch item
         * shared memory = max seq len for this batch
         */
        int max_seq_len =
            *std::max_element(h_seq_lens.begin(), h_seq_lens.end());

        size_t smem = (size_t)max_seq_len * sizeof(float);

        dim3 grid(N_HEADS, B);

        paged_decode_mha_batch_kernel<<<grid, DECODE_BLOCK, smem>>>(
            buf_ln,
            d_batch_block_tables,
            d_seq_lens,
            pool,
            buf_O,
            B,
            max_blocks,
            block_size,
            D_MODEL,
            D_HEAD,
            scale
        );

        CUDA_CHECK(cudaGetLastError());

        /*
         * 2-7. Output projection
         * buf_O    : [B, D_MODEL]
         * buf_attn : [B, D_MODEL]
         */
        linear_wmma(
            buf_O,
            W.out_w[l],
            W.out_b[l],
            buf_attn,
            B,
            D_MODEL,
            D_MODEL,
            buf_xh
        );

        /*
         * Residual connection:
         * buf_x = buf_x + attention_out
         */
        residual_add(
            buf_x,
            buf_attn,
            B * D_MODEL
        );

        /*
         * 2-8. FFN
         */
        layernorm(
            buf_x,
            buf_ln,
            W.ln2_w[l],
            W.ln2_b[l],
            B,
            D_MODEL
        );

        linear_wmma(
            buf_ln,
            W.fc1_w[l],
            W.fc1_b[l],
            buf_ff,
            B,
            D_MODEL,
            D_FF,
            buf_xh
        );

        gelu(
            buf_ff,
            B * D_FF
        );

        linear_wmma(
            buf_ff,
            W.fc2_w[l],
            W.fc2_b[l],
            buf_attn,
            B,
            D_FF,
            D_MODEL,
            buf_xh
        );

        /*
         * Residual connection:
         * buf_x = buf_x + ffn_out
         */
        residual_add(
            buf_x,
            buf_attn,
            B * D_MODEL
        );
    }

    /*
     * 3. Final LayerNorm
     */
    layernorm(
        buf_x,
        buf_ln,
        W.ln_f_w,
        W.ln_f_b,
        B,
        D_MODEL
    );

    /*
     * 4. LM head
     *
     * buf_ln: [B, D_MODEL]
     * W.wte : [VOCAB, D_MODEL]
     * logits: [B, VOCAB]
     */
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        VOCAB,
        B,
        D_MODEL,
        &alpha,
        W.wte,
        D_MODEL,
        buf_ln,
        D_MODEL,
        &beta,
        buf_batch_logits,
        VOCAB
    ));

    /*
     * 5. Copy logits to host and argmax per request.
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_logits((size_t)B * VOCAB);

    CUDA_CHECK(cudaMemcpy(
        h_logits.data(),
        buf_batch_logits,
        (size_t)B * VOCAB * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    std::vector<int> next_tokens(B);

    for (int b = 0; b < B; b++) {
        const float* row = h_logits.data() + (size_t)b * VOCAB;

        next_tokens[b] =
            (int)(std::max_element(row, row + VOCAB) - row);
    }

    return next_tokens;
}
