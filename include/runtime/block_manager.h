#pragma once

#include <vector>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include "runtime/block.h"
#include "runtime/pool.cuh"
#include "runtime/cpu_pool.cuh"


namespace mini_llm::runtime {

class BlockManager {
private:
    Pool& pool_;
    CpuPool& cpu_pool_;

    int total_blocks_;
    int free_blocks_;

    int total_cpu_blocks_;
    int free_cpu_blocks_;

    std::vector<PhysicalBlock> blocks_;
    std::vector<int> free_block_ids_;
    std::vector<bool> is_free_;

    std::vector<int> free_cpu_block_ids_;
    std::vector<bool> is_cpu_free_;

    
    explicit BlockManager(Pool& pool, CpuPool& cpu_pool)
        : pool_(pool),
          total_blocks_(pool.total_blocks()),
          free_blocks_(pool.total_blocks()),
          is_free_(pool.total_blocks(), true),
          cpu_pool_(cpu_pool),
          total_cpu_blocks_(cpu_pool.total_blocks()),
          free_cpu_blocks_(cpu_pool.total_blocks()),
          is_cpu_free_(cpu_pool_.total_blocks(), true)
           {
        blocks_.reserve(total_blocks_);
        free_block_ids_.reserve(total_blocks_);

        for (int i = 0; i < total_blocks_; i++) {
            blocks_.emplace_back(i);
            free_block_ids_.push_back(i);
        }

        free_cpu_block_ids_.reserve(total_cpu_blocks_);
        for(int i = 0; i < total_cpu_blocks_; i++) free_cpu_block_ids_.push_back(i);

    }

public:
    static BlockManager& getInstance(Pool& pool, CpuPool& cpu_pool) {
        static BlockManager instance(pool, cpu_pool);
        return instance;
    }

    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;

    // --- GPU Pool 관련 method들 ---
    bool can_allocate(int n) const;

    int free_blocks_num() const;

    int total_blocks() const;

    int allocate_one();

    std::vector<int> allocate(int n);

    void free_one(int block_id);

    void free(const std::vector<int>& block_ids);

    PhysicalBlock& block(int block_id);

    const PhysicalBlock& block(int block_id) const;

    float* block_ptr(int block_id);

    Pool& pool();

    // --- CPU pool 관련 메소드들 ---

    int free_cpu_blocks_num() const;

    int allocate_cpu_one();

    std::vector<int> allocate_cpu(int n);

    void free_cpu_one(int cpu_block_id);

    void free_cpu(const std::vector<int>& cpu_block_ids);

    void move_data(int block_id, int cpu_block_id, bool is_move_gpu_to_cpu);
};

} // namespace mini_llm::runtime