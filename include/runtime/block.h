#pragma once
#include <cstdio>
#include "constants.h"
namespace mini_llm::runtime {

class PhysicalBlock {
private:
    int id_;
    int filled_ = 0;
    int refcount_ = 0;
public:
    PhysicalBlock(int id)
        : id_(id) {}

    int id() const { return id_; }
    int filled() const { return filled_; }
    bool full() const { return filled_ >= mini_llm::constants::DEFAULT_KV_BLOCK_SIZE; }

    int reserve_one_token() {
        if (full()) return -1;
        return filled_++;
    }

    int reserve_tokens(int n) {
        int start = filled_;
        filled_ += n;
        return start;
    }

    void reset() {
        filled_ = 0;
        refcount_ = 0;
    }
};

}