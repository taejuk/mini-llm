#pragma once

#include "runtime/kv_allocator.h"
#include "runtime/block_manager.h"
#include "runtime/swapdata.h"
#include "runtime/pool.cuh"
#include "runtime/cpu_pool.cuh"
#include "constants.h"

#include <unordered_map>

namespace mini_llm::runtime {

class RealKvAllocator final : public KvAllocator {
private:
    Pool& pool_;
    CpuPool& cpu_pool_;
    BlockManager& block_manager_;
    std::unordered_map<uint64_t, SwappedRequest> swapped_requests_;

public:
    RealKvAllocator()
        : pool_(Pool::getInstance(mini_llm::constants::DEFAULT_TOTAL_KV_BLOCKS)),
          cpu_pool_(CpuPool::getInstance(mini_llm::constants::DEFAULT_TOTAL_CPU_BLOCKS)),
          block_manager_(BlockManager::getInstance(pool_, cpu_pool_)) {}

    bool allocate_prefill(Request& req) override;
    bool allocate_decode(Request& req) override;
    void free_request(Request& req) override;

    bool swap_out(Request& req) override;
    bool swap_in(Request& req) override;
    bool is_swapped(const Request& req) const override;
};

} // namespace mini_llm::runtime