#pragma once

#include "vllm/allocator.cuh"

#include <vector>
#include <utility>
#include <cstdio>
#include <cstdlib>
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

    /*
     * Device-side block table.
     *
     * 기존에는 append_qkv_from_interleaved() 안에서 매번
     * cudaMalloc -> cudaMemcpy -> cudaFree를 했다.
     *
     * 이제는 PagedKVCache가 d_block_table을 계속 들고 있고,
     * reserve_blocks_for_tokens()에서 한 번 준비한 뒤 재사용한다.
     */
    int* d_block_table;
    int  d_block_table_capacity;

    void ensure_device_block_table_capacity(int needed_blocks);
    void upload_new_block_ids_to_device(int old_blocks, const std::vector<int>& new_phys_ids);
    void update_filled_counts();

public:
    PagedKVCache(int b_size, int h_dim, int s_id)
        : block_size(b_size),
          hidden_dim(h_dim),
          seq_id(s_id),
          block_table(),
          num_tokens(0),
          d_block_table(nullptr),
          d_block_table_capacity(0) {}

    ~PagedKVCache();

    /*
     * d_block_table raw pointer가 있으므로 copy는 막는다.
     * std::vector<PagedKVCache>에서 이동은 가능하게 둔다.
     */
    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    PagedKVCache(PagedKVCache&& other) noexcept
        : block_size(other.block_size),
          hidden_dim(other.hidden_dim),
          seq_id(other.seq_id),
          block_table(std::move(other.block_table)),
          num_tokens(other.num_tokens),
          d_block_table(other.d_block_table),
          d_block_table_capacity(other.d_block_table_capacity) {
        other.d_block_table = nullptr;
        other.d_block_table_capacity = 0;
        other.num_tokens = 0;
    }

    PagedKVCache& operator=(PagedKVCache&& other) noexcept {
        if (this != &other) {
            if (d_block_table) {
                cudaFree(d_block_table);
            }

            block_size = other.block_size;
            hidden_dim = other.hidden_dim;
            seq_id = other.seq_id;
            block_table = std::move(other.block_table);
            num_tokens = other.num_tokens;
            d_block_table = other.d_block_table;
            d_block_table_capacity = other.d_block_table_capacity;

            other.d_block_table = nullptr;
            other.d_block_table_capacity = 0;
            other.num_tokens = 0;
        }

        return *this;
    }

    /*
     * prefill 전에 호출:
     * prompt_len이 정해지면 필요한 block 수가 정해지므로
     * 여기서 physical block과 d_block_table을 미리 준비한다.
     */
    void reserve_blocks_for_tokens(int total_tokens);

    /*
     * 기존 append 함수.
     * decode 등에서 안전하게 사용할 수 있도록 필요한 block이 없으면 할당한다.
     */
    void append_token_kv(float* new_k, float* new_v);

    void append_token_kv_batch(float* new_k, float* new_v, int n);

    /*
     * 기존 API는 유지.
     * 내부에서 reserve 후 no_alloc append를 호출한다.
     */
    void append_qkv_from_interleaved(const float* buf_qkv, int seq_len);

    /*
     * prefill 최적화용:
     * 이미 reserve_blocks_for_tokens()가 호출되었다고 가정하고,
     * 여기서는 cudaMalloc/cudaMemcpy 없이 K/V write kernel만 실행한다.
     */
    void append_qkv_from_interleaved_no_alloc(const float* buf_qkv, int seq_len);

    PagedKVCache fork();

    void cow_append(float* key_vec, float* value_vec);

    void free_all();

    void append_block_table_entry(const BlockTableEntry& bte);

    void set_num_tokens(int t) { num_tokens = t; }

    int get_seq_id() const { return seq_id; }
    int get_num_tokens() const { return num_tokens; }
    int get_num_blocks() const { return (int)block_table.size(); }

    const std::vector<BlockTableEntry>& get_block_table() const {
        return block_table;
    }

    const int* get_device_block_table() const {
        return d_block_table;
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
