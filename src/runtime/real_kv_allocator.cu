#include "runtime/real_kv_allocator.h"
#include <cuda_runtime.h>


namespace mini_llm::runtime {
bool RealKvAllocator::allocate_prefill(Request& req) override {
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

bool RealKvAllocator::allocate_decode(Request& req) override {
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
    // swap out할만큼 block이 충분한지 확인한다.
    int needed_cpu_blocks = req->layer_kv[0].block_table_.size() * mini_llm::constants::GPT2_N_LAYERS;

    if(needed_cpu_blocks > block_manager_.free_cpu_blocks_num()) return false;
    std::vector<int> allocated_cpu_blocks = block_manager_.allocate_cpu(needed_cpu_blocks);
    std::vector<SwappedBlock> swapped_blocks;
    int idx = 0;
    for(int i = 0; i < mini_llm::constants::GPT2_N_LAYERS; i++) {
        for(int gpu_block_id : req->layer_kv[i].block_table_) {
            int cpu_block_id = allocated_cpu_blocks[idx];
            block_manager_.move_data(gpu_block_id, cpu_block_id, true);
            swapped_blocks.emplace_back(cpu_block_id, block_manager_.block(gpu_block_id).filled);
            idx++;
        }
    }

    SwappedRequest swapped_req = SwappedRequest{
        swapped_blocks, req->layer_kv[0].size()
    };
    swapped_requests_[req->request_id] = swapped_req;
    // 나중에 request 상태를 업데이트한다. 
    free_request(req);
    req->kv_residency = KvCacheResidency::Cpu;
    return true;
}

bool RealKvAllocator::swap_in(Request& req) {
    SwappedRequest swapped_req = swapped_requests_[req->request_id];
    int blocks_per_layer = swapped_req.blocks_per_layer;
    // gpu에 공간이 있는지 확인해야 한다.
    if(!block_manager_.can_allocate(blocks_per_layer * mini_llm::constants::GPT2_N_LAYERS)) return false;
    for(int i = 0; i < mini_llm::constants::GPT2_N_LAYERS; i++) {
        vector<int> block_tables;
        for(int j = 0; j < blocks_per_layer) {
            // gpu block 할당받고
            // 
        }
    }
}

bool RealKvAllocator::is_swapped(const Request& req) const {
    
}

}