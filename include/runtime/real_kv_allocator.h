#pragma once

#include "runtime/kv_allocator.h"
#include "runtime/block_manager.h"
#include "runtime/pool.cuh"
#include "constants.h"

namespace mini_llm::runtime {

class RealKvAllocator final : public KvAllocator {
private:
    Pool& pool_;
    BlockManager& block_manager_;

public:
    RealKvAllocator()
        : pool_(Pool::getInstance(mini_llm::constants::DEFAULT_TOTAL_KV_BLOCKS)),
          block_manager_(BlockManager::getInstance(pool_)) {}

    bool allocate_prefill(Request& req) override {
        namespace C = mini_llm::constants;

        int need_blocks_per_layer =
            (req.prompts_len + C::DEFAULT_KV_BLOCK_SIZE - 1) /
            C::DEFAULT_KV_BLOCK_SIZE;

        int total_need = need_blocks_per_layer * C::GPT2_N_LAYERS;

        if (!block_manager_.can_allocate(total_need)) {
            return false;
        }

        for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
            for (int i = 0; i < need_blocks_per_layer; i++) {
                int block_id = block_manager_.allocate_one();
                req.layer_kv[layer].append_block(block_id);
            }
        }

        return true;
    }

    bool allocate_decode(Request& req) override {
        namespace C = mini_llm::constants;

        int cached_tokens = req.layer_kv[0].num_tokens_;

        int need_blocks_per_layer =
            (cached_tokens % C::DEFAULT_KV_BLOCK_SIZE) == 0 ? 1 : 0;

        int total_need = need_blocks_per_layer * C::GPT2_N_LAYERS;

        if (!block_manager_.can_allocate(total_need)) {
            return false;
        }

        for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
            for (int i = 0; i < need_blocks_per_layer; i++) {
                int block_id = block_manager_.allocate_one();
                req.layer_kv[layer].append_block(block_id);
            }
        }

        return true;
    }

    void free_request(Request& req) override {
        for (auto& kv : req.layer_kv) {
            block_manager_.free(kv.block_table_);
            kv.reset();
        }
    }
};

} // namespace mini_llm::runtime