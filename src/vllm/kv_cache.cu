#include "vllm/kv_cache.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void append_qkv_to_paged_kv_kernel(
    const float* __restrict__ buf_qkv,       // [seq_len, 3 * hidden_dim]
    float* __restrict__ kv_pool,             // BlockAllocator pool
    const int* __restrict__ block_table,     // logical block -> physical block id
    int start_pos,
    int seq_len,
    int block_size,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = seq_len * hidden_dim;
    if (idx >= total) return;

    int s = idx / hidden_dim;  // 이번 append 안에서 몇 번째 token
    int d = idx % hidden_dim;  // hidden dimension index

    int pos = start_pos + s;

    int logical_block = pos / block_size;
    int slot          = pos % block_size;
    int phys_id       = block_table[logical_block];

    size_t block_elems = (size_t)block_size * hidden_dim;
    size_t base        = (size_t)phys_id * 2 * block_elems;

    const float* k_src = buf_qkv
        + (size_t)s * 3 * hidden_dim
        + hidden_dim
        + d;

    const float* v_src = buf_qkv
        + (size_t)s * 3 * hidden_dim
        + 2 * hidden_dim
        + d;

    float* k_dst = kv_pool
        + base
        + (size_t)slot * hidden_dim
        + d;

    float* v_dst = kv_pool
        + base
        + block_elems
        + (size_t)slot * hidden_dim
        + d;

    *k_dst = *k_src;
    *v_dst = *v_src;
}

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

void PagedKVCache::append_qkv_from_interleaved(
    const float* buf_qkv,
    int seq_len
) {
    if (seq_len <= 0) return;

    int start_pos = num_tokens;
    int end_pos   = num_tokens + seq_len;

    int needed_blocks  = (end_pos + block_size - 1) / block_size;
    int current_blocks = (int)block_table.size();

    while ((int)block_table.size() < needed_blocks) {
        int phys_id     = BlockAllocator::getInstance().allocate();
        int logical_idx = (int)block_table.size();

        block_table.push_back({
            logical_idx,
            phys_id,
            0
        });
    }

    std::vector<int> h_phys(block_table.size());

    for (int i = 0; i < (int)block_table.size(); i++) {
        h_phys[i] = block_table[i].phys_block_id;
    }

    int* d_block_table = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_block_table,
        h_phys.size() * sizeof(int)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_block_table,
        h_phys.data(),
        h_phys.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    float* kv_pool = BlockAllocator::getInstance().get_pool();

    int total = seq_len * hidden_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    append_qkv_to_paged_kv_kernel<<<blocks, threads>>>(
        buf_qkv,
        kv_pool,
        d_block_table,
        start_pos,
        seq_len,
        block_size,
        hidden_dim
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_block_table));

    num_tokens = end_pos;

    for (auto& entry : block_table) {
        int block_start = entry.logical_idx * block_size;
        int block_end   = block_start + block_size;

        int filled = num_tokens - block_start;

        if (filled < 0) {
            filled = 0;
        } else if (filled > block_size) {
            filled = block_size;
        }

        entry.filled = filled;
    }
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
