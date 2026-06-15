#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <iostream>

#include "constants.h"

namespace mini_llm::runtime {

class CpuPool {
private:
    int total_blocks_ = 0;
    float* pool_ = nullptr;

    explicit CpuPool(int total_blocks)
        : total_blocks_(total_blocks),
          pool_(nullptr) {
        if (total_blocks_ <= 0) {
            std::cerr << "CpuPool: total_blocks must be positive, got "
                      << total_blocks_ << "\n";
            std::exit(1);
        }

        size_t bytes =
            static_cast<size_t>(total_blocks_) *
            block_slot_size() *
            sizeof(float);

        cudaError_t err = cudaMallocHost(
            reinterpret_cast<void**>(&pool_),
            bytes
        );

        if (err != cudaSuccess) {
            std::cerr << "CpuPool: cudaMallocHost failed: "
                      << cudaGetErrorString(err) << "\n";
            std::exit(1);
        }
    }

public:

    static CpuPool& getInstance(
        int t_blocks = mini_llm::constants::DEFAULT_TOTAL_CPU_BLOCKS
    ) {
        static CpuPool instance(t_blocks);
        return instance;
    }

    ~CpuPool() {
        if (pool_ != nullptr) {
            cudaError_t err = cudaFreeHost(pool_);
            if (err != cudaSuccess) {
                std::cerr << "CpuPool: cudaFreeHost failed: "
                          << cudaGetErrorString(err) << "\n";
            }

            pool_ = nullptr;
        }
    }

    CpuPool(const CpuPool&) = delete;
    CpuPool& operator=(const CpuPool&) = delete;

    CpuPool(CpuPool&&) = delete;
    CpuPool& operator=(CpuPool&&) = delete;

    int total_blocks() const {
        return total_blocks_;
    }

    float* pool_start() const {
        return pool_;
    }

    size_t block_slot_size() const {
        return
            static_cast<size_t>(mini_llm::constants::DEFAULT_KV_BLOCK_SIZE) *
            static_cast<size_t>(mini_llm::constants::GPT2_D_MODEL) *
            2;
    }

    size_t block_slot_bytes() const {
        return block_slot_size() * sizeof(float);
    }

    size_t total_bytes() const {
        return
            static_cast<size_t>(total_blocks_) *
            block_slot_bytes();
    }

    float* block_ptr(int block_id) const {
        assert(block_id >= 0);
        assert(block_id < total_blocks_);

        return pool_ + static_cast<size_t>(block_id) * block_slot_size();
    }
};

} // namespace mini_llm::runtime