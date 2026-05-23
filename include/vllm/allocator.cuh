#pragma once

#include "vllm/block.h"
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
// singleton 패턴으로 해야 함. 이유: pool은 하나여야하기 때문
class BlockAllocator {
private:
    int total_blocks;
    int block_size;
    int hidden_dim;
    float* pool;                      
    std::vector<PhysicalBlock> blocks;
    std::vector<int> free_blocks;

    BlockAllocator(int t_blocks, int b_size, int h_dim)
        : total_blocks(t_blocks), block_size(b_size), hidden_dim(h_dim)
    {
        
        size_t slot = (size_t)b_size * h_dim * 2 * sizeof(float); 
        cudaMalloc(&pool, (size_t)t_blocks * slot);

        float* cur = pool;
        for (int i = 0; i < t_blocks; i++) {
            blocks.push_back(PhysicalBlock(b_size, h_dim, cur, cur + b_size * h_dim));
            cur += b_size * h_dim * 2; 
        }
        for (int i = t_blocks - 1; i >= 0; i--) free_blocks.push_back(i);
    }

public:
    static BlockAllocator& getInstance(int t_blocks = 0,
                                       int b_size   = 0,
                                       int h_dim    = 0)
    {
        static BlockAllocator instance(t_blocks, b_size, h_dim);
        return instance;
    }
    BlockAllocator(const BlockAllocator&)            = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    int allocate() {
        if (free_blocks.empty()) {
            fprintf(stderr, "BlockAllocator::allocate: OOM\n");
            exit(1);
        }
        int id = free_blocks.back();
        free_blocks.pop_back();
        blocks[id].inc_ref();
        return id;
    }

    void free(int block_id) {
        check_range("free", block_id);
        PhysicalBlock& b = blocks[block_id];
        if (b.get_refcount() == 0) {
            fprintf(stderr, "BlockAllocator::free: double free(%d)\n", block_id);
            exit(1);
        }
        b.dec_ref();
        if (b.get_refcount() == 0) {
            b.reset();
            free_blocks.push_back(block_id);
        }
    }

    void inc_ref(int block_id) {
        check_range("inc_ref", block_id);
        blocks[block_id].inc_ref();
    }

    void dec_ref(int block_id) {
        check_range("dec_ref", block_id);
        blocks[block_id].dec_ref();
    }

    int copy_block(int src_block_id, int filled) {
        check_range("copy_block", src_block_id);
        PhysicalBlock& src = blocks[src_block_id];
        int dst_id = allocate();
        PhysicalBlock& dst = blocks[dst_id];

        size_t sz = (size_t)filled * hidden_dim * sizeof(float);
        cudaMemcpy(dst.get_key(),   src.get_key(),   sz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dst.get_value(), src.get_value(), sz, cudaMemcpyDeviceToDevice);
        return dst_id;
    }

    PhysicalBlock& get_block(int block_id) {
        check_range("get_block", block_id);
        return blocks[block_id];
    }

    float* get_pool() const { return pool; }

    int get_block_size()       const { return block_size; }
    int get_hidden_dim()       const { return hidden_dim; }
    int get_num_free_blocks()  const { return (int)free_blocks.size(); }
    int get_total_blocks()     const { return total_blocks; }

    void print_stats() const {
        printf("BlockAllocator: %d / %d blocks free\n",
               (int)free_blocks.size(), total_blocks);
    }

private:
    void check_range(const char* fn, int id) const {
        if (id < 0 || id >= (int)blocks.size()) {
            fprintf(stderr, "BlockAllocator::%s: out of range(%d)\n", fn, id);
            exit(1);
        }
    }
};
