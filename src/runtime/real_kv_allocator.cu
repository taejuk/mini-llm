#include "runtime/real_kv_allocator.h"
#include <cuda_runtime.h>


namespace mini_llm::runtime {
bool RealKvAllocator::allocate_prefill(Request& req) {
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

bool RealKvAllocator::allocate_decode(Request& req) {
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

void RealKvAllocator::free_request(Request& req) override {
    for (auto& kv : req.layer_kv) {
        block_manager_.free(kv.block_table_);
        kv.reset();
    }
}

bool RealKvAllocator::swap_out(Request& req) {
    namespace C = mini_llm::constants;

    int blocks_per_layer = req.layer_kv[0].num_blocks();
    int needed_cpu_blocks = blocks_per_layer * C::GPT2_N_LAYERS;

    if (needed_cpu_blocks > block_manager_.free_cpu_blocks_num()) {
        return false;
    }

    std::vector<int> allocated_cpu_blocks =
        block_manager_.allocate_cpu(needed_cpu_blocks);

    if (static_cast<int>(allocated_cpu_blocks.size()) != needed_cpu_blocks) {
        return false;
    }

    std::vector<SwappedBlock> swapped_blocks;
    swapped_blocks.reserve(needed_cpu_blocks);

    int idx = 0;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        for (int gpu_block_id : req.layer_kv[layer].block_table_) {
            int cpu_block_id = allocated_cpu_blocks[idx++];

            block_manager_.move_data(
                gpu_block_id,
                cpu_block_id,
                true    // GPU -> CPU
            );

            swapped_blocks.push_back({
                cpu_block_id,
                block_manager_.block(gpu_block_id).filled()
            });
        }
    }

    SwappedRequest swapped_req{
        std::move(swapped_blocks),
        blocks_per_layer
    };

    swapped_requests_[req.request_id] = std::move(swapped_req);

    free_request(req);

    req.kv_residency = KvCacheResidency::Cpu;

    return true;
}

bool RealKvAllocator::swap_in(Request& req) {
    namespace C = mini_llm::constants;

    auto it = swapped_requests_.find(req.request_id);
    if (it == swapped_requests_.end()) {
        return false;
    }

    SwappedRequest& swapped_req = it->second;
    int blocks_per_layer = swapped_req.blocks_per_layer;

    int expected_blocks =
        blocks_per_layer * C::GPT2_N_LAYERS;

    if (static_cast<int>(swapped_req.cpu_blocks.size()) != expected_blocks) {
        return false;
    }

    if (static_cast<int>(req.layer_kv.size()) != C::GPT2_N_LAYERS) {
        return false;
    }

    if (!block_manager_.can_allocate(expected_blocks)) {
        return false;
    }

    int restored_num_tokens = 0;

    for (int b = 0; b < blocks_per_layer; ++b) {
        restored_num_tokens += swapped_req.cpu_blocks[b].valid_tokens;
    }

    int idx = 0;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];
        kv.reset();

        for (int b = 0; b < blocks_per_layer; ++b) {
            const SwappedBlock& swapped_block = swapped_req.cpu_blocks[idx++];

            int gpu_block_id = block_manager_.allocate_one();
            if (gpu_block_id < 0) {
                return false;
            }

            block_manager_.move_data(
                gpu_block_id,
                swapped_block.cpu_block_id,
                false   // CPU -> GPU
            );

            block_manager_.block(gpu_block_id)
                .reserve_tokens(swapped_block.valid_tokens);

            kv.append_block(gpu_block_id);

            block_manager_.free_cpu_one(swapped_block.cpu_block_id);
        }

        kv.set_num_tokens(restored_num_tokens);
    }

    swapped_requests_.erase(it);

    req.kv_residency = KvCacheResidency::Gpu;
    req.state = RequestState::DecodeReady;

    return true;
}

bool RealKvAllocator::is_swapped(const Request& req) const {
    return swapped_requests_.find(req.request_id) != swapped_requests_.end();
}

}