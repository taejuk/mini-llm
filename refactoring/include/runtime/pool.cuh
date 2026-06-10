#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#include "constants.h"

namespace mini_llm::runtime {

class Pool {
private:
    int total_blocks_;
    float* pool_;

    explicit Pool(int t_blocks)
        : total_blocks_(t_blocks), pool_(nullptr) {
        size_t slot =
            static_cast<size_t>(mini_llm::constants::DEFAULT_KV_BLOCK_SIZE) *
            mini_llm::constants::GPT2_D_MODEL *
            2 *
            sizeof(float);

        cudaError_t err = cudaMalloc(&pool_, slot * static_cast<size_t>(t_blocks));
        if (err != cudaSuccess) {
            std::cerr << "Pool: cudaMalloc failed: "
                      << cudaGetErrorString(err) << "\n";
            std::exit(1);
        }
    }

public:
    static Pool& getInstance(int t_blocks = 0) {
        static Pool instance(t_blocks);
        return instance;
    }

    ~Pool() {
        if (pool_ != nullptr) {
            cudaFree(pool_);
            pool_ = nullptr;
        }
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    float* pool_start() const {
        return pool_;
    }

    int total_blocks() const {
        return total_blocks_;
    }
};

} // namespace mini_llm::runtime