#pragma once

#include "vllm/allocator.cuh"
#include <vector>
#include <cuda_runtime.h>

struct BlockTableEntry {
    int logical_idx;   
    int phys_block_id; 
    int filled;
};


class PagedKVCache {
private:
    int block_size;
    int hidden_dim;
    int seq_id;
    std::vector<BlockTableEntry> block_table;
    int num_tokens;

public:
    PagedKVCache(int b_size, int h_dim, int s_id)
        : block_size(b_size), hidden_dim(h_dim),
          seq_id(s_id), num_tokens(0) {}

    void append_token_kv(float* new_k, float* new_v);

    void append_token_kv_batch(float* new_k, float* new_v, int n);

    PagedKVCache fork();

    void cow_append(float* key_vec, float* value_vec);

    void free_all();

    void append_block_table_entry(const BlockTableEntry& bte) {
        block_table.push_back(bte);
    }
    void set_num_tokens(int t) { num_tokens = t; }

    int    get_seq_id()     const { return seq_id; }
    int    get_num_tokens() const { return num_tokens; }
    int    get_num_blocks() const { return (int)block_table.size(); }

    const std::vector<BlockTableEntry>& get_block_table() const {
        return block_table;
    }

    size_t get_memory_usage() const {
        return (size_t)block_table.size() * block_size * hidden_dim * 2 * sizeof(float);
    }

    size_t get_wasted_memory() const {
        if (block_table.empty()) return 0;
        size_t wasted_slots = (size_t)block_table.size() * block_size - num_tokens;
        return wasted_slots * hidden_dim * 2 * sizeof(float);
    }
};
