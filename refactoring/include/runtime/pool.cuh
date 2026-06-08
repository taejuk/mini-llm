#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include "runtime/block.h"
#include "constants.h"

namespace mini_llm::runtime {
class Pool {
private:
    int total_blocks_;
    bool init_;
    float* pool_;
    PhysicalBlock* head_ = nullptr;
    Pool(int t_blocks)
    : total_blocks(t_blocks){
        size_t slot = (size_t)mini_llm::constants::DEFAULT_KV_BLOCK_SIZE * mini_llm::constants::GPT2_D_MODEL * 2 * sizeof(float);
        cudaMalloc(&pool_, slot * (size_t)t_blocks);
        PhysicalBlock* tail;
        for(int i = 0; i < t_blocks; i++) {
            // head를 연결하고, new를 통해서 block을 생성해야 한다,
            PhysicalBlock* newblock = new PhysicalBlock(i);
            if(head_ == nullptr) head_ = newblock;
            else tail->set_next(newblock);
            tail = newblock;
        }
    }

public:
    static Pool& getInstance(int t_blocks = 0) {
        static Pool instance(t_blocks);
        return instance;
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;


    PhysicalBlock* getBlock() {
        if(head_==nullptr) {
            std::cerr << "Pool: No block\n";
            exit(1);
        }
        PhysicalBlock* block = head_;
        head_ = head_->get_next();
        block->set_next(nullptr);
        return block;
    }

    void free(PhysicalBlock* block) {
        if(head_ == nullptr) {
            head_ = block;
            return;
        }
        block->set_next(head_);
        head_ = block;
    }
};
}