#pragma once

namespace mini_llm::constants {

inline constexpr int GPT2_N_LAYERS = 12;
inline constexpr int GPT2_N_HEADS = 12;
inline constexpr int GPT2_D_MODEL = 768;
inline constexpr int GPT2_D_HEAD = GPT2_D_MODEL / GPT2_N_HEADS;
inline constexpr int GPT2_VOCAB_SIZE = 50257;
inline constexpr int GPT2_D_FF = 3072;

inline constexpr int DEFAULT_PORT = 8080;
inline constexpr int DEFAULT_BACKLOG = 128;
inline constexpr int DEFAULT_MAX_NEW_TOKENS = 8;

inline constexpr int DEFAULT_KV_BLOCK_SIZE = 16;
inline constexpr int DEFAULT_TOTAL_KV_BLOCKS = 4096;

inline constexpr int DEFAULT_MAX_DECODE_BATCH_SIZE = 8;
inline constexpr int DEFAULT_MAX_PREFILL_PER_ITER = 1;
inline constexpr int DEFAULT_REQUEST_POOL_SIZE = 1024;

inline constexpr int MAX_SEQ = 1024;
inline constexpr char* WEIGHTS_DIR = "weights";
inline constexpr int MAX_BATCH_NUM = 16;

}