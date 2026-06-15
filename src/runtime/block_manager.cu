#include "runtime/block_manager.h"

#include <cuda_runtime.h>

namespace mini_llm::runtime {

bool BlockManager::can_allocate(int n) const {
    return n >= 0 && free_blocks_ >= n;
}

int BlockManager::free_blocks_num() const {
    return free_blocks_;
}

int BlockManager::total_blocks() const {
    return total_blocks_;
}

PhysicalBlock& BlockManager::block(int block_id) {
    assert(block_id >= 0 && block_id < total_blocks_);
    return blocks_[block_id];
}

const PhysicalBlock& BlockManager::block(int block_id) const {
    assert(block_id >= 0 && block_id < total_blocks_);
    return blocks_[block_id];
}

float* BlockManager::block_ptr(int block_id) {
    assert(block_id >= 0 && block_id < total_blocks_);
    assert(!is_free_[block_id]);
    return pool_.block_ptr(block_id);
}

Pool& BlockManager::pool() {
    return pool_;
}

int BlockManager::allocate_one() {
    if (free_block_ids_.empty()) {
        return -1;
    }

    int block_id = free_block_ids_.back();
    free_block_ids_.pop_back();

    assert(block_id >= 0 && block_id < total_blocks_);
    assert(is_free_[block_id]);

    is_free_[block_id] = false;
    free_blocks_--;

    blocks_[block_id].reset();

    return block_id;
}

std::vector<int> BlockManager::allocate(int n) {
    std::vector<int> ret;

    if (!can_allocate(n)) {
        return ret;
    }

    ret.reserve(n);

    for (int i = 0; i < n; i++) {
        int block_id = allocate_one();
        assert(block_id >= 0);
        ret.push_back(block_id);
    }

    return ret;
}

void BlockManager::free_one(int block_id) {
    if (block_id < 0 || block_id >= total_blocks_) {
        std::cerr << "BlockManager: invalid block id "
                  << block_id << "\n";
        return;
    }

    if (is_free_[block_id]) {
        std::cerr << "BlockManager: double free block id "
                  << block_id << "\n";
        return;
    }

    blocks_[block_id].reset();
    is_free_[block_id] = true;
    free_block_ids_.push_back(block_id);
    free_blocks_++;
}

void BlockManager::free(const std::vector<int>& block_ids) {
    for (int block_id : block_ids) {
        free_one(block_id);
    }
}

int BlockManager::free_cpu_blocks_num() const {
    return free_cpu_blocks_;
}

int BlockManager::allocate_cpu_one() {
    if (free_cpu_block_ids_.empty()) {
        return -1;
    }

    int id = free_cpu_block_ids_.back();
    free_cpu_block_ids_.pop_back();

    assert(id >= 0 && id < total_cpu_blocks_);
    assert(is_cpu_free_[id]);

    free_cpu_blocks_--;
    is_cpu_free_[id] = false;

    return id;
}

std::vector<int> BlockManager::allocate_cpu(int n) {
    std::vector<int> ids;

    if (n < 0 || free_cpu_blocks_ < n) {
        return ids;
    }

    ids.reserve(n);

    for (int i = 0; i < n; i++) {
        int id = allocate_cpu_one();
        if (id < 0) {
            free_cpu(ids);
            ids.clear();
            return ids;
        }

        ids.push_back(id);
    }

    return ids;
}

void BlockManager::free_cpu_one(int cpu_block_id) {
    if (cpu_block_id < 0 || cpu_block_id >= total_cpu_blocks_) {
        std::cerr << "BlockManager: invalid cpu block id "
                  << cpu_block_id << "\n";
        return;
    }

    if (is_cpu_free_[cpu_block_id]) {
        std::cerr << "BlockManager: double free cpu block id "
                  << cpu_block_id << "\n";
        return;
    }

    is_cpu_free_[cpu_block_id] = true;
    free_cpu_block_ids_.push_back(cpu_block_id);
    free_cpu_blocks_++;
}

void BlockManager::free_cpu(const std::vector<int>& cpu_block_ids) {
    for (int id : cpu_block_ids) {
        free_cpu_one(id);
    }
}

bool BlockManager::move_data(
    int block_id,
    int cpu_block_id,
    bool is_move_gpu_to_cpu
) {
    float* gpu_addr = pool_.block_ptr(block_id);
    float* cpu_addr = cpu_pool_.block_ptr(cpu_block_id);
    size_t size = pool_.block_slot_size() * sizeof(float);

    cudaError_t err;

    if (is_move_gpu_to_cpu) {
        err = cudaMemcpy(cpu_addr, gpu_addr, size, cudaMemcpyDeviceToHost);
    } else {
        err = cudaMemcpy(gpu_addr, cpu_addr, size, cudaMemcpyHostToDevice);
    }

    if (err != cudaSuccess) {
        std::cerr << "BlockManager::move_data cudaMemcpy failed: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    return true;
}

} // namespace mini_llm::runtime
