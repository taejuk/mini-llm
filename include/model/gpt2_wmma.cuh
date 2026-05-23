#pragma once

#include "model/gpt2_common.cuh"
#include "kernel/wmma_gemm.cuh"
#include "vllm/kv_cache.cuh"
#include "scheduler/request.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

#define MAX_BATCH 32   // 동시 처리 최대 시퀀스 수

/* ============================================================
 * GPT2WeightsWMMA
 *
 * 선형 변환 가중치: FP16 + 전치(Transpose) 저장
 *   wmma_gemm(A, B, C) = C = A @ B  이므로
 *   y = x @ W^T 계산 시 B = W^T 를 미리 저장
 *
 * 원본 shape   → 저장 shape
 *   qkv_w [3D, D] → [D,  3D]
 *   out_w [D,  D] → [D,  D ]
 *   fc1_w [FF, D] → [D,  FF]
 *   fc2_w [D, FF] → [FF, D ]
 * ============================================================ */
struct GPT2WeightsWMMA {
    /* FP32 — 임베딩 */
    float* wte;                // [VOCAB, D_MODEL]
    float* wpe;                // [MAX_SEQ, D_MODEL]

    /* FP32 — Layer Norm */
    float* ln1_w[N_LAYERS];
    float* ln1_b[N_LAYERS];
    float* ln2_w[N_LAYERS];
    float* ln2_b[N_LAYERS];
    float* ln_f_w;
    float* ln_f_b;

    /* FP32 — Bias */
    float* qkv_b[N_LAYERS];   // [3*D_MODEL]
    float* out_b[N_LAYERS];   // [D_MODEL]
    float* fc1_b[N_LAYERS];   // [D_FF]
    float* fc2_b[N_LAYERS];   // [D_MODEL]

    /* FP16, 전치 — 선형 변환 가중치 */
    __half* qkv_w[N_LAYERS];  // [D_MODEL,  3*D_MODEL]
    __half* out_w[N_LAYERS];  // [D_MODEL,  D_MODEL  ]
    __half* fc1_w[N_LAYERS];  // [D_MODEL,  D_FF     ]
    __half* fc2_w[N_LAYERS];  // [D_FF,     D_MODEL  ]
};

/* ============================================================
 * GPT2ModelWMMA — Singleton
 *
 * - 선형 변환       : wmma_gemm (FP16 Tensor Core)
 * - Prefill 어텐션  : cuBLAS StridedBatched (FP32)
 * - Decode 어텐션   : paged_decode_mha 커널 (단일 / 배치)
 * ============================================================ */
class GPT2ModelWMMA {
public:
    static void           init(const char* weight_dir, int blk_size);
    static GPT2ModelWMMA& get();

    GPT2ModelWMMA(const GPT2ModelWMMA&)            = delete;
    GPT2ModelWMMA& operator=(const GPT2ModelWMMA&) = delete;

    /* 단일 시퀀스 */
    int prefill(const int* d_token_ids, int prompt_len, PagedKVCache& kv);
    int decode_step(int token_id, PagedKVCache& kv);

    /* 배치 decode — Scheduler의 batch.decode_reqs를 직접 받음
     * 반환: 각 요청의 next token id (reqs와 동일한 순서) */
    std::vector<int> batch_decode(const std::vector<Request*>& reqs);

    float* get_logits() const { return buf_logits; }

private:
    GPT2WeightsWMMA W;
    cublasHandle_t  handle;

    /* 공용 스크래치 버퍼 */
    float*  buf_x;        // [MAX_SEQ, D_MODEL]
    float*  buf_ln;       // [MAX_SEQ, D_MODEL]
    float*  buf_qkv;      // [MAX_SEQ, 3*D_MODEL]
    __half* buf_xh;       // [MAX_SEQ, D_FF]  — FP16 변환 스크래치
    float*  buf_S;        // [N_HEADS, MAX_SEQ, MAX_SEQ]
    float*  buf_O;        // [MAX_SEQ, D_MODEL]
    float*  buf_attn;     // [MAX_SEQ, D_MODEL]
    float*  buf_ff;       // [MAX_SEQ, D_FF]
    float*  buf_logits;   // [VOCAB]

    /* 단일 decode 전용 */
    int* d_block_table;   // [MAX_SEQ/block_size + 1]

    /* batch_decode 전용 */
    int*   d_batch_block_tables; // [MAX_BATCH * (MAX_SEQ/block_size + 1)]
    int*   d_seq_lens;           // [MAX_BATCH]
    float* buf_batch_logits;     // [MAX_BATCH * VOCAB]

    int block_size;

    GPT2ModelWMMA(const char* weight_dir, int blk_size);
};
