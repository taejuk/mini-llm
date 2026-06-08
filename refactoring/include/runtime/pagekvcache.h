#pragma once

#include <vector>
#include <cassert>

#include "constants.h"

namespace mini_llm::runtime {

struct PagedKVCache {
    int request_id_ = -1;
    std::vector<int> block_table_;
    int num_tokens_ = 0;

    explicit PagedKVCache(int req_id)
        : request_id_(req_id),
          num_tokens_(0) {}

    int num_blocks() const {
        return static_cast<int>(block_table_.size());
    }

    int physical_block_id(int logical_block_id) const {
        assert(logical_block_id >= 0);
        assert(logical_block_id < static_cast<int>(block_table_.size()));
        return block_table_[logical_block_id];
    }

    void append_block(int physical_block_id) {
        block_table_.push_back(physical_block_id);
    }

    void set_num_tokens(int n) {
        num_tokens_ = n;
    }

    void increment_tokens(int n = 1) {
        num_tokens_ += n;
    }

    void reset() {
        block_table_.clear();
        num_tokens_ = 0;
    }
};

} // namespace mini_llm::runtime