#include "runtime/real_kv_allocator.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <utility>
#include <vector>
#include <limits>

namespace mini_llm::runtime {
namespace C = mini_llm::constants;

__global__ void scatter_swap_in_staging_kernel(
    const float* __restrict__ staging,
    float* __restrict__ gpu_pool,
    const int* __restrict__ dst_block_ids,
    int num_blocks,
    size_t block_elems
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elems =
        static_cast<size_t>(num_blocks) * block_elems;

    if (tid >= total_elems) {
        return;
    }

    int staging_block = static_cast<int>(tid / block_elems);
    size_t elem_offset = tid % block_elems;

    int dst_block_id = dst_block_ids[staging_block];

    gpu_pool[
        static_cast<size_t>(dst_block_id) * block_elems + elem_offset
    ] = staging[tid];
}

RealKvAllocator::~RealKvAllocator() {
    release_swap_in_buffers();
}

bool RealKvAllocator::init_swap_in_buffers() {
    namespace C = mini_llm::constants;

    int max_blocks_per_layer =
        (C::MAX_SEQ + C::DEFAULT_KV_BLOCK_SIZE - 1) /
        C::DEFAULT_KV_BLOCK_SIZE;

    int max_total_blocks =
        max_blocks_per_layer * C::GPT2_N_LAYERS;

    size_t block_elems = pool_.block_slot_size();

    size_t staging_bytes =
        static_cast<size_t>(max_total_blocks) *
        block_elems *
        sizeof(float);

    cudaError_t err = cudaMalloc(
        reinterpret_cast<void**>(&d_swap_in_staging_),
        staging_bytes
    );

    if (err != cudaSuccess) {
        std::cerr << "RealKvAllocator::init_swap_in_buffers "
                  << "cudaMalloc staging failed: "
                  << cudaGetErrorString(err) << "\n";
        d_swap_in_staging_ = nullptr;
        swap_in_capacity_blocks_ = 0;
        return false;
    }

    err = cudaMalloc(
        reinterpret_cast<void**>(&d_swap_in_dst_block_ids_),
        static_cast<size_t>(max_total_blocks) * sizeof(int)
    );

    if (err != cudaSuccess) {
        std::cerr << "RealKvAllocator::init_swap_in_buffers "
                  << "cudaMalloc dst ids failed: "
                  << cudaGetErrorString(err) << "\n";

        cudaFree(d_swap_in_staging_);
        d_swap_in_staging_ = nullptr;
        d_swap_in_dst_block_ids_ = nullptr;
        swap_in_capacity_blocks_ = 0;
        return false;
    }

    swap_in_capacity_blocks_ = max_total_blocks;

    return true;
}

void RealKvAllocator::release_swap_in_buffers() {
    if (d_swap_in_staging_ != nullptr) {
        cudaFree(d_swap_in_staging_);
        d_swap_in_staging_ = nullptr;
    }

    if (d_swap_in_dst_block_ids_ != nullptr) {
        cudaFree(d_swap_in_dst_block_ids_);
        d_swap_in_dst_block_ids_ = nullptr;
    }

    swap_in_capacity_blocks_ = 0;

}



bool RealKvAllocator::allocate_prefill(Request& req) {
    

    int need_blocks_per_layer =
        (req.prompts_len + C::DEFAULT_KV_BLOCK_SIZE - 1) /
        C::DEFAULT_KV_BLOCK_SIZE;

    int total_need = need_blocks_per_layer * C::GPT2_N_LAYERS;

    if (!block_manager_.can_allocate(total_need)) {
        return false;
    }

    for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        for (int i = 0; i < need_blocks_per_layer; i++) {
            int block_id = block_manager_.allocate_one();
            if (block_id < 0) {
                return false;
            }

            req.layer_kv[layer].append_block(block_id);
        }
    }

    req.kv_residency = KvCacheResidency::Gpu;

    return true;
}

bool RealKvAllocator::allocate_decode(Request& req) {

    int cached_tokens = req.layer_kv[0].num_tokens_;

    int need_blocks_per_layer =
        (cached_tokens % C::DEFAULT_KV_BLOCK_SIZE) == 0 ? 1 : 0;

    int total_need = need_blocks_per_layer * C::GPT2_N_LAYERS;

    if (!block_manager_.can_allocate(total_need)) {
        return false;
    }

    for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        for (int i = 0; i < need_blocks_per_layer; i++) {
            int block_id = block_manager_.allocate_one();
            if (block_id < 0) {
                return false;
            }

            req.layer_kv[layer].append_block(block_id);
        }
    }

    req.kv_residency = KvCacheResidency::Gpu;

    return true;
}

void RealKvAllocator::free_request(Request& req) {
    for (auto& kv : req.layer_kv) {
        block_manager_.free(kv.block_table_);
        kv.reset();
    }

    req.kv_residency = KvCacheResidency::None;
}

bool RealKvAllocator::swap_out(Request& req) {

    if (req.layer_kv.empty()) {
        return false;
    }

    int blocks_per_layer = req.layer_kv[0].num_blocks();
    int needed_cpu_blocks = blocks_per_layer * C::GPT2_N_LAYERS;

    if (needed_cpu_blocks <= 0) {
        return false;
    }

    if (needed_cpu_blocks > block_manager_.free_cpu_blocks_num()) {
        return false;
    }

    std::vector<int> allocated_cpu_blocks =
        block_manager_.allocate_cpu(needed_cpu_blocks);

    if (static_cast<int>(allocated_cpu_blocks.size()) != needed_cpu_blocks) {
        return false;
    }

    std::vector<SwappedBlock> swapped_blocks;
    swapped_blocks.reserve(needed_cpu_blocks);

    int idx = 0;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];

        if (kv.num_blocks() != blocks_per_layer) {
            block_manager_.free_cpu(allocated_cpu_blocks);
            return false;
        }

        for (int b = 0; b < kv.num_blocks(); ++b) {
            int gpu_block_id = kv.block_table_[b];
            int cpu_block_id = allocated_cpu_blocks[idx++];

            if (!block_manager_.move_data(
                    gpu_block_id,
                    cpu_block_id,
                    true    // GPU -> CPU
                )) {
                block_manager_.free_cpu(allocated_cpu_blocks);
                return false;
            }

            int block_start = b * C::DEFAULT_KV_BLOCK_SIZE;
            int remain = kv.num_tokens_ - block_start;
            int valid_tokens = std::max(
                0,
                std::min(remain, C::DEFAULT_KV_BLOCK_SIZE)
            );

            swapped_blocks.push_back({
                cpu_block_id,
                valid_tokens
            });
        }
    }

    SwappedRequest swapped_req{
        std::move(swapped_blocks),
        blocks_per_layer
    };

    swapped_requests_[req.request_id] = std::move(swapped_req);

    free_request(req);

    req.kv_residency = KvCacheResidency::Cpu;
    req.state = RequestState::SwappedOut;

    return true;
}


// cpu에는 연속적으로 저장되니깐 (swap_out은 하나만 호출하기 때문에 걱정할 필요없다.)
// buffer_에 바로 넣으면 된다.
bool RealKvAllocator::swap_in(Request& req) {

    auto it = swapped_requests_.find(req.request_id);

    if (it == swapped_requests_.end()) {
        return false;
    }

    SwappedRequest& swapped_req = it->second;
    int blocks_per_layer = swapped_req.blocks_per_layer;
    int expected_blocks = blocks_per_layer * C::GPT2_N_LAYERS;

    if (expected_blocks <= 0) {
        return false;
    }

    if (static_cast<int>(swapped_req.cpu_blocks.size()) != expected_blocks) {
        return false;
    }

    if (static_cast<int>(req.layer_kv.size()) != C::GPT2_N_LAYERS) {
        return false;
    }

    if (!block_manager_.can_allocate(expected_blocks)) {
        return false;
    }

    if (d_swap_in_staging_ == nullptr ||
        d_swap_in_dst_block_ids_ == nullptr ||
        swap_in_capacity_blocks_ <= 0) {
        std::cerr
            << "RealKvAllocator::swap_in: staging buffers are not initialized\n";
        return false;
    }

    if (expected_blocks > swap_in_capacity_blocks_) {
        std::cerr
            << "RealKvAllocator::swap_in: expected_blocks="
            << expected_blocks
            << " exceeds staging capacity="
            << swap_in_capacity_blocks_
            << "\n";
        return false;
    }

    int cpu_start = swapped_req.cpu_blocks[0].cpu_block_id;

    for (const SwappedBlock& swapped_block : swapped_req.cpu_blocks) {
        cpu_start = std::min(cpu_start, swapped_block.cpu_block_id);
    }

    int restored_num_tokens = 0;

    for (int b = 0; b < blocks_per_layer; ++b) {
        restored_num_tokens += swapped_req.cpu_blocks[b].valid_tokens;
    }

    std::vector<int> restored_gpu_blocks;
    restored_gpu_blocks.reserve(expected_blocks);

    std::vector<int> dst_block_by_cpu_offset(
        expected_blocks,
        -1
    );

    auto rollback_gpu_allocations = [&]() {
        block_manager_.free(restored_gpu_blocks);

        for (auto& kv : req.layer_kv) {
            kv.reset();
        }
    };

    int idx = 0;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];
        kv.reset();

        for (int b = 0; b < blocks_per_layer; ++b) {
            const SwappedBlock& swapped_block =
                swapped_req.cpu_blocks[idx++];

            int gpu_block_id = block_manager_.allocate_one();

            if (gpu_block_id < 0) {
                rollback_gpu_allocations();
                return false;
            }

            restored_gpu_blocks.push_back(gpu_block_id);

            int cpu_offset = swapped_block.cpu_block_id - cpu_start;

            dst_block_by_cpu_offset[cpu_offset] = gpu_block_id;

            block_manager_.block(gpu_block_id)
                .reserve_tokens(swapped_block.valid_tokens);

            kv.append_block(gpu_block_id);
        }

        kv.set_num_tokens(restored_num_tokens);
    }

    size_t block_elems = pool_.block_slot_size();

    size_t total_bytes =
        static_cast<size_t>(expected_blocks) *
        block_elems *
        sizeof(float);

    cudaError_t err = cudaMemcpy(
        d_swap_in_staging_,
        cpu_pool_.block_ptr(cpu_start),
        total_bytes,
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        std::cerr
            << "RealKvAllocator::swap_in staging H2D failed: "
            << cudaGetErrorString(err)
            << "\n";

        rollback_gpu_allocations();
        return false;
    }

    err = cudaMemcpy(
        d_swap_in_dst_block_ids_,
        dst_block_by_cpu_offset.data(),
        static_cast<size_t>(expected_blocks) * sizeof(int),
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        std::cerr
            << "RealKvAllocator::swap_in dst ids H2D failed: "
            << cudaGetErrorString(err)
            << "\n";

        rollback_gpu_allocations();
        return false;
    }

    size_t total_elems =
        static_cast<size_t>(expected_blocks) * block_elems;

    constexpr int threads = 256;

    int grid = static_cast<int>(
        (total_elems + threads - 1) / threads
    );

    scatter_swap_in_staging_kernel<<<grid, threads>>>(
        d_swap_in_staging_,
        pool_.pool_start(),
        d_swap_in_dst_block_ids_,
        expected_blocks,
        block_elems
    );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cerr
            << "RealKvAllocator::swap_in scatter launch failed: "
            << cudaGetErrorString(err)
            << "\n";

        rollback_gpu_allocations();
        return false;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr
            << "RealKvAllocator::swap_in scatter failed: "
            << cudaGetErrorString(err)
            << "\n";

        rollback_gpu_allocations();
        return false;
    }

    for (const SwappedBlock& swapped_block : swapped_req.cpu_blocks) {
        block_manager_.free_cpu_one(swapped_block.cpu_block_id);
    }

    swapped_requests_.erase(it);

    req.kv_residency = KvCacheResidency::Gpu;
    req.state = RequestState::DecodeReady;

    return true;
}

bool RealKvAllocator::is_swapped(const Request& req) const {
    return swapped_requests_.find(req.request_id) != swapped_requests_.end();
}

} // namespace mini_llm::runtime
