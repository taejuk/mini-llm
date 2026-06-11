#pragma once

#include <vector>
#include <cassert>
#include <iostream>

#include "runtime/block.h"
#include "runtime/pool.cuh"

namespace mini_llm::runtime {

class BlockManager {
private:
    Pool& pool_;
    int total_blocks_;
    int free_blocks_;

    std::vector<PhysicalBlock> blocks_;
    std::vector<int> free_block_ids_;
    std::vector<bool> is_free_;

    explicit BlockManager(Pool& pool)
        : pool_(pool),
          total_blocks_(pool.total_blocks()),
          free_blocks_(pool.total_blocks()),
          is_free_(pool.total_blocks(), true) {
        blocks_.reserve(total_blocks_);
        free_block_ids_.reserve(total_blocks_);

        for (int i = 0; i < total_blocks_; i++) {
            blocks_.emplace_back(i);
            free_block_ids_.push_back(i);
        }
    }

public:
    static BlockManager& getInstance(Pool& pool) {
        static BlockManager instance(pool);
        return instance;
    }

    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;

    bool can_allocate(int n) const {
        return n >= 0 && free_blocks_ >= n;
    }

    int free_blocks_num() const {
        return free_blocks_;
    }

    int total_blocks() const {
        return total_blocks_;
    }

    int allocate_one() {
        if (free_block_ids_.empty()) {
            return -1;
        }

        int block_id = free_block_ids_.back();
        free_block_ids_.pop_back();

        assert(block_id >= 0 && block_id < total_blocks_);
        assert(is_free_[block_id]);

        is_free_[block_id] = false;
        free_blocks_--;

        blocks_[block_id].reset();

        return block_id;
    }

    std::vector<int> allocate(int n) {
        std::vector<int> ret;

        if (!can_allocate(n)) {
            return ret;
        }

        ret.reserve(n);

        for (int i = 0; i < n; i++) {
            int block_id = allocate_one();
            assert(block_id >= 0);
            ret.push_back(block_id);
        }

        return ret;
    }

    void free_one(int block_id) {
        if (block_id < 0 || block_id >= total_blocks_) {
            std::cerr << "BlockManager: invalid block id "
                      << block_id << "\n";
            return;
        }

        if (is_free_[block_id]) {
            std::cerr << "BlockManager: double free block id "
                      << block_id << "\n";
            return;
        }

        blocks_[block_id].reset();
        is_free_[block_id] = true;
        free_block_ids_.push_back(block_id);
        free_blocks_++;
    }

    void free(const std::vector<int>& block_ids) {
        for (int block_id : block_ids) {
            free_one(block_id);
        }
    }

    PhysicalBlock& block(int block_id) {
        assert(block_id >= 0 && block_id < total_blocks_);
        return blocks_[block_id];
    }

    const PhysicalBlock& block(int block_id) const {
        assert(block_id >= 0 && block_id < total_blocks_);
        return blocks_[block_id];
    }

    float* block_ptr(int block_id) {
        assert(block_id >= 0 && block_id < total_blocks_);
        assert(!is_free_[block_id]);
        return pool_.block_ptr(block_id);
    }

    Pool& pool() {
        return pool_;
    }
};

} // namespace mini_llm::runtime