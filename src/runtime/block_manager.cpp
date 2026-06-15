#include "runtime/block_manager.h"

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

const BlockManager::PhysicalBlock& block(int block_id) const {
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

// GPU pool 관련 메소스들
int BlockManager::free_cpu_blocks_num() const {
    return free_cpu_blocks_;
}

int BlockManager::allocate_cpu_one() {
    if(free_cpu_blocks_ < 1) return -1;
    int id = free_cpu_block_ids_.back();
    free_cpu_block_ids_.pop_back();
    free_cpu_blocks_--;
    is_cpu_free_[id] = false;
    return id;
}

std::vector<int> BlockManager::allocate_cpu(int n) {
    if(free_cpu_blocks_ < n) return {-1};
    std::vector<int> ids;
    for(int i = 0; i < n; i++) ids.push_back(allocate_cpu_one());
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
    for(int id : cpu_block_ids) free_cpu_one(id);
}


}