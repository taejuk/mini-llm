#pragma once

#include "runtime/block.h"

#include <vector>

namespace mini_llm::runtime {

struct BlockTableEntry {
    int logical_id;
    PhysicalBlock* physicalblock;
};

class PagedKVCache {
private:
    int req_id_;
    std::vector<BlockTableEntry> block_table_;
    int num_tokens_;
    
public:
    PagedKVCache(int req_id) : req_id_(req_id) {}
    
    ~PagedKVCache();
    void append_kv_prefill(float* buf_qkv, int seq_len);
    void append_kv_decode(float* new_k, float* new_v);
};

}