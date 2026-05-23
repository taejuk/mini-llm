#pragma once

#include "vllm/kv_cache.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <cstdint>

/* ── GPT-2 small 상수 ──────────────────────────────────────────── */
#ifndef D_MODEL
#define D_MODEL   768
#define N_HEADS   12
#define D_HEAD    64
#define N_LAYERS  12
#define D_FF      3072
#define VOCAB     50257
#define MAX_SEQ   1024
#endif


struct LinearArgs {
    __half*       tmp     = nullptr;   
    int8_t*       tmp_i8  = nullptr;   
    const float*  scales  = nullptr;  
};

/* ── 공용 커널 wrappers (gpt2_common.cu에 정의) ─────────────────── */

// Layer Normalization: y = w * (x - mean) / std + b  [seq_len, d]
void layernorm(const float* x, float* y,
               const float* w, const float* b,
               int seq_len, int d);

// GELU activation (in-place)  x: [n]
void gelu(float* x, int n);

// Bias add: y[i] += b[i % out_feat]
void bias_add(float* y, const float* b, int seq_len, int out_feat);

// Residual: x[i] += r[i]
void residual_add(float* x, const float* r, int n);

// Token + Position Embedding
void embedding_lookup(const int* d_token_ids, const float* wte,
                      const float* wpe, float* x,
                      int seq_len, int d_model, int pos_offset);

// Causal mask: S[h][row][col] = -1e9 if col > row
void causal_mask_apply(float* S, int seq_len, int n_heads);

// Row-wise softmax over S[n_heads, seq_len, seq_len]
void softmax_prefill(float* S, int seq_len, int n_heads);

// FP32 -> FP16 변환 (GPU 내부)
void float_to_half_device(const float* src, __half* dst, int n);

/* ============================================================
 * linear<W_T> — 가중치 자료형 템플릿
 *
 * y = x @ W^T + b
 *   x : [seq_len, in_feat]  FP32
 *   W : [out_feat, in_feat] W_T
 *   b : [out_feat]          FP32 (nullable)
 *   y : [seq_len, out_feat] FP32
 *
 * W_T = float   → cublasSgemm (FP32 cuBLAS)
 * W_T = __half  → cublasGemmEx CUBLAS_COMPUTE_32F_FAST_16F (TC)
 * W_T = int8_t  → (TODO) cublasLtMatmul INT8 IMMA + dequant
 * ============================================================ */
template<typename W_T>
inline void linear(cublasHandle_t h,
                   const float* x, const W_T* W, const float* b, float* y,
                   int seq_len, int in_feat, int out_feat,
                   LinearArgs args = {})
{
    /* ── FP32 ─────────────────────────────────────────────────── */
    if constexpr (std::is_same_v<W_T, float>) {
        const float alpha = 1.f, beta = 0.f;
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
            out_feat, seq_len, in_feat,
            &alpha, W, in_feat, x, in_feat,
            &beta,  y, out_feat);
    }
    /* ── FP16 Tensor Core ─────────────────────────────────────── */
    else if constexpr (std::is_same_v<W_T, __half>) {
        // x(FP32) → tmp(FP16)
        float_to_half_device(x, args.tmp, seq_len * in_feat);

        const float alpha = 1.f, beta = 0.f;
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
            out_feat, seq_len, in_feat,
            &alpha,
            W,         CUDA_R_16F, in_feat,
            args.tmp,  CUDA_R_16F, in_feat,
            &beta,
            y,         CUDA_R_32F, out_feat,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    /* ── INT8 IMMA (TODO) ─────────────────────────────────────── */
    else if constexpr (std::is_same_v<W_T, int8_t>) {
        // 구현 단계:
        //   1) x(FP32) → args.tmp_i8(INT8)  quantize 커널 (per-tensor scale)
        //   2) cublasLtMatmul: CUDA_R_8I input, CUDA_R_32I output
        //      ComputeType = CUBLAS_COMPUTE_32I (IMMA)
        //   3) dequant: y[i] = y_i32[i] * args.scales[i % out_feat]
        static_assert(!std::is_same_v<W_T, W_T>,  // always fires when instantiated
            "INT8 IMMA not yet implemented. Use cublasLtMatmul + per-channel dequant.");
        (void)args;
    }

    // Bias 공통 처리
    if (b) bias_add(y, b, seq_len, out_feat);
}
