#pragma once

#include "runtime/kv_allocator.h"
#include "constants.h"

namespace mini_llm::runtime {

class MockKvAllocator final : public KvAllocator {
private:
    int next_block_id_ = 0;

public:
    bool allocate_prefill(Request& req) override {
        namespace C = mini_llm::constants;

        int blocks_per_layer =
            (req.prompts_len + C::DEFAULT_KV_BLOCK_SIZE - 1) /
            C::DEFAULT_KV_BLOCK_SIZE;

        for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
            for (int i = 0; i < blocks_per_layer; i++) {
                req.layer_kv[layer].append_block(next_block_id_++);
            }
        }

        return true;
    }

    bool allocate_decode(Request& req) override {
        namespace C = mini_llm::constants;

        int cached_tokens = req.layer_kv[0].num_tokens_;
        bool need_new_block =
            (cached_tokens % C::DEFAULT_KV_BLOCK_SIZE) == 0;

        if (!need_new_block) {
            return true;
        }

        for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
            req.layer_kv[layer].append_block(next_block_id_++);
        }

        return true;
    }

    void free_request(Request& req) override {
        for (auto& kv : req.layer_kv) {
            kv.reset();
        }
    }

    bool swap_out(Request& req) {
        return true;
    }

    bool swap_in(Request& req) {
        return true;
    }

    bool is_swapped(const Request& req) const {
        return true;
    }
};

} // namespace mini_llm::runtime