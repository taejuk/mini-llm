#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#include "runtime/pagekvcache.h"
#include "constants.h"

namespace mini_llm::runtime {

struct ClientConnection;

enum class RequestState {
    WaitingPrefill,
    DeferredPrefillKV,
    PrefillRunning,

    DecodeReady,
    DeferredDecodeKV,
    DecodeRunning,

    Finished,
    Aborted
};

struct Request {
    uint64_t request_id = static_cast<uint64_t>(-1);
    std::vector<int> tokens;
    int prompts_len = 0;

    std::vector<PagedKVCache> layer_kv;

    int max_new_tokens = 8;
    RequestState state = RequestState::WaitingPrefill;

    int* d_block_tables_ = nullptr;
    size_t offset = 0;

    Request() = default;

    Request(int req_id, std::vector<int> t, int max_tokens = 8)
        : request_id(req_id),
          tokens(std::move(t)),
          prompts_len(static_cast<int>(tokens.size())),
          max_new_tokens(max_tokens),
          state(RequestState::WaitingPrefill) {
        for (int i = 0; i < mini_llm::constants::GPT2_N_LAYERS; i++) {
            layer_kv.emplace_back(req_id);
        }

        size_t len_per_pagedkv =
            (tokens.size() + max_tokens +
             mini_llm::constants::DEFAULT_KV_BLOCK_SIZE - 1)
            / mini_llm::constants::DEFAULT_KV_BLOCK_SIZE;

        offset = len_per_pagedkv * sizeof(int);
    }

    int generated_tokens() const {
        return static_cast<int>(tokens.size()) - prompts_len;
    }

    bool isfinish() const {
        return generated_tokens() >= max_new_tokens;
    }
};

} // namespace mini_llm::runtime