#include "vllm/kv_cache.cuh"

void PagedKVCache::append_token_kv(float* new_k, float* new_v) {
    if (block_table.empty() || block_table.back().filled >= block_size) {
        int phys_id     = BlockAllocator::getInstance().allocate();
        int logical_idx = (int)block_table.size();
        block_table.push_back({logical_idx, phys_id, 0});
    }

    BlockTableEntry& entry = block_table.back();
    PhysicalBlock&   block = BlockAllocator::getInstance().get_block(entry.phys_block_id);

    size_t sz = (size_t)hidden_dim * sizeof(float);
    cudaMemcpy(block.get_key_at(entry.filled),   new_k, sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(block.get_value_at(entry.filled),  new_v, sz, cudaMemcpyDeviceToDevice);

    entry.filled++;
    num_tokens++;
}

void PagedKVCache::append_token_kv_batch(float* new_k, float* new_v, int n) {
    for (int i = 0; i < n; i++)
        append_token_kv(new_k + (size_t)i * hidden_dim,
                        new_v + (size_t)i * hidden_dim);
}

PagedKVCache PagedKVCache::fork() {
    PagedKVCache new_cache(block_size, hidden_dim, /*seq_id=*/-1);

    for (const BlockTableEntry& entry : block_table) {
        BlockAllocator::getInstance().inc_ref(entry.phys_block_id);
        new_cache.append_block_table_entry(entry);   // 동일 물리 블록 공유
    }
    new_cache.set_num_tokens(num_tokens);
    return new_cache;
}

void PagedKVCache::cow_append(float* key_vec, float* value_vec) {
    if (!block_table.empty()) {
        BlockTableEntry& last = block_table.back();
        PhysicalBlock& blk   = BlockAllocator::getInstance().get_block(last.phys_block_id);

        if (blk.get_refcount() > 1 && last.filled < block_size) {
            int new_id = BlockAllocator::getInstance().copy_block(
                             last.phys_block_id, last.filled);
            BlockAllocator::getInstance().dec_ref(last.phys_block_id);
            last.phys_block_id = new_id;
        }
    }
    append_token_kv(key_vec, value_vec);
}

void PagedKVCache::free_all() {
    for (const BlockTableEntry& entry : block_table)
        BlockAllocator::getInstance().free(entry.phys_block_id);
    block_table.clear();
    num_tokens = 0;
}
