#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "runtime/pagekvcache.h"
#include "constants.h"


namespace mini_llm::runtime {
struct ClientConnection;

enum class RequestState {
    WaitingPrefill,
    DeferredPrefillKV,  // prefill 시작 전에 KV 부족
    PrefillRunning,

    DecodeReady,
    DeferredDecodeKV,   // decode 도중 KV 부족
    DecodeRunning,

    Finished,
    Aborted
};


struct Request {
    uint64_t request_id = -1;
    std::vector<int> tokens;
    std::vector<PagedKVCache> layer_kv;
    int max_new_tokens = 8;
    RequestState state;
    // gpu에 저장해야하는 blocktableentry.
    // pool 시작점과 block_id_만 주면 되니깐 int로 저장해도 된다.
    int* d_block_tables_;
    size_t offset;
    Request() = default;

    Request(int req_id, std::vector<int> t, int max_tokens = 8)
        : request_id(req_id),
          tokens(std::move(t)),
          max_new_tokens(max_tokens),
          state(RequestState::WaitingPrefill)
          {
            for(int i = 0; i < mini_llm::constants::GPT2_N_LAYERS; i++) layer_kv.emplace_back(req_id);
            // d_block_tables_e도 초기화해야 함.
            size_t len_per_pagedkv = (t.size() + max_tokens + mini_llm::constants::DEFAULT_KV_BLOCK_SIZE - 1) / mini_llm::constants::DEFAULT_KV_BLOCK_SIZE;
            offset = len_per_pagedkv * sizeof(int);
          }
    int required_blocks() {
        int needed_per_layer = (tokens.size() + mini_llm::constants::DEFAULT_KV_BLOCK_SIZE - 1) / mini_llm::constants::DEFAULT_KV_BLOCK_SIZE;
        return needed_per_layer * mini_llm::constants::GPT2_N_LAYERS;
    }
};
}