#include "model/gpt2_common.cuh"

/* ============================================================
 * 공용 CUDA 커널 정의
 * gpt2_model.cu / gpt2_fp16_tc.cu 양쪽에서 include 없이
 * 링크로 공유. 각 모델은 gpt2_common.cuh만 include.
 * ============================================================ */

/* ── layernorm ──────────────────────────────────────────────── */
__global__ void layernorm_kernel(const float* x, float* y,
                                 const float* w, const float* b,
                                 int d, float eps)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x, row = blockIdx.x;
    const float* rx = x + (size_t)row * d;
    float*       ry = y + (size_t)row * d;

    /* mean */
    float sum = 0.f;
    for (int i = tid; i < d; i += blockDim.x) sum += rx[i];
    smem[tid] = sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s]; __syncthreads();
    }
    float mean = smem[0] / d;

    /* variance */
    float var = 0.f;
    for (int i = tid; i < d; i += blockDim.x) {
        float d_ = rx[i] - mean; var += d_ * d_;
    }
    smem[tid] = var; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s]; __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / d + eps);
    for (int i = tid; i < d; i += blockDim.x)
        ry[i] = w[i] * (rx[i] - mean) * inv_std + b[i];
}

void layernorm(const float* x, float* y,
               const float* w, const float* b, int seq_len, int d) {
    layernorm_kernel<<<seq_len, 256, 256*sizeof(float)>>>(x, y, w, b, d, 1e-5f);
}

/* ── GELU ───────────────────────────────────────────────────── */
__global__ void gelu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f*v*v*v)));
    }
}

void gelu(float* x, int n) {
    gelu_kernel<<<(n+255)/256, 256>>>(x, n);
}

/* ── Bias Add ───────────────────────────────────────────────── */
__global__ void bias_add_kernel(float* y, const float* b, int n, int out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % out];
}

void bias_add(float* y, const float* b, int seq_len, int out_feat) {
    bias_add_kernel<<<(seq_len*out_feat+255)/256, 256>>>(y, b, seq_len*out_feat, out_feat);
}

/* ── Residual Add ───────────────────────────────────────────── */
__global__ void residual_kernel(float* x, const float* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += r[i];
}

void residual_add(float* x, const float* r, int n) {
    residual_kernel<<<(n+255)/256, 256>>>(x, r, n);
}

/* ── Embedding Lookup ───────────────────────────────────────── */
__global__ void embedding_kernel(const int* ids, const float* wte,
                                 const float* wpe, float* x,
                                 int seq_len, int d, int pos_offset)
{
    int pos = blockIdx.x, dim = threadIdx.x;
    if (pos >= seq_len || dim >= d) return;
    x[(size_t)pos*d + dim] = wte[(size_t)ids[pos]*d + dim]
                           + wpe[(size_t)(pos_offset+pos)*d + dim];
}

void embedding_lookup(const int* d_token_ids, const float* wte,
                      const float* wpe, float* x,
                      int seq_len, int d_model, int pos_offset) {
    embedding_kernel<<<seq_len, d_model>>>(
        d_token_ids, wte, wpe, x, seq_len, d_model, pos_offset);
}

/* ── Causal Mask ────────────────────────────────────────────── */
__global__ void causal_mask_kernel(float* S, int seq_len) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= seq_len) return;
    /* blockIdx.z = head index */
    float* rs = S + (size_t)blockIdx.z * seq_len * seq_len;
    if (col > row) rs[(size_t)row * seq_len + col] = -1e9f;
}

void causal_mask_apply(float* S, int seq_len, int n_heads) {
    dim3 grid((seq_len+31)/32, seq_len, n_heads);
    causal_mask_kernel<<<grid, 32>>>(S, seq_len);
}

/* ── Softmax (row-wise over S[n_heads, seq_len, seq_len]) ───── */
__global__ void softmax_kernel(float* S, int seq_len) {
    int head = blockIdx.y, row = blockIdx.x;
    int tid  = threadIdx.x, bs = blockDim.x;
    float* rs = S + (size_t)head * seq_len * seq_len + (size_t)row * seq_len;
    __shared__ float smem[32];

    /* row max */
    float lm = -INFINITY;
    for (int j = tid; j < seq_len; j += bs) lm = fmaxf(lm, rs[j]);
    smem[tid] = lm; __syncthreads();
    for (int s = bs/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]); __syncthreads();
    }
    float row_max = smem[0]; __syncthreads();

    /* exp & sum */
    float ls = 0.f;
    for (int j = tid; j < seq_len; j += bs) {
        float e = expf(rs[j] - row_max); rs[j] = e; ls += e;
    }
    smem[tid] = ls; __syncthreads();
    for (int s = bs/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s]; __syncthreads();
    }
    float inv = 1.f / smem[0]; __syncthreads();
    for (int j = tid; j < seq_len; j += bs) rs[j] *= inv;
}

void softmax_prefill(float* S, int seq_len, int n_heads) {
    dim3 grid(seq_len, n_heads);
    softmax_kernel<<<grid, 32>>>(S, seq_len);
}

/* ── FP32 → FP16 변환 ───────────────────────────────────────── */
__global__ void float_to_half_kernel(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void float_to_half_device(const float* src, __half* dst, int n) {
    float_to_half_kernel<<<(n+255)/256, 256>>>(src, dst, n);
}
