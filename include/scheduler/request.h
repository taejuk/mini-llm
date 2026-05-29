#pragma once

#include "vllm/kv_cache.cuh"
#include "model/gpt2_common.cuh"
#include <vector>
#include <future>

enum class RequestState {
    WAITING,   // 큐에서 대기 중
    PREFILL,   // 이번 step에 prefill 예정
    DECODE,    // 토큰 생성 중
    DONE,      // 생성 완료
};

struct Request {
    int                           id;
    std::vector<int>              prompt_ids;
    std::vector<int>              output_ids;
    int                           max_new_tokens;
    RequestState                  state;
    std::vector<PagedKVCache>                  layer_kv;

    std::promise<std::vector<int>> result_promise;

    Request(int id, std::vector<int> prompt, int max_tokens, int block_size, int hidden_dim)
        : id(id)
        , prompt_ids(std::move(prompt))
        , max_new_tokens(max_tokens)
        , state(RequestState::WAITING)
    {
    	layer_kv.reserve(N_LAYERS);

	for (int l = 0; l < N_LAYERS; l++) {
            layer_kv.emplace_back(block_size, hidden_dim, id);
        }
    }

    Request(const Request&)            = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&)                 = default;
};
