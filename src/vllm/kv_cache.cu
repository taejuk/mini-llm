#include "vllm/kv_cache.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <nvtx3/nvToolsExt.h>

#ifndef KV_APPEND_VEC_WIDTH
#define KV_APPEND_VEC_WIDTH 4
#endif

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

/*
 * Baseline scalar kernel.
 *
 * thread 하나가:
 *   K[s, d] 1개
 *   V[s, d] 1개
 * 를 복사한다.
 */
__global__ void append_qkv_to_paged_kv_vec1_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ kv_pool,
    const int* __restrict__ block_table,
    int start_pos,
    int seq_len,
    int block_size,
    int hidden_dim
) {
    int s = blockIdx.x;
    if (s >= seq_len) return;

    __shared__ int shared_phys_id;
    __shared__ int shared_slot;

    if (threadIdx.x == 0) {
        int pos = start_pos + s;

        int logical_block = pos / block_size;
        int slot          = pos % block_size;
        int phys_id       = block_table[logical_block];

        shared_phys_id = phys_id;
        shared_slot    = slot;
    }

    __syncthreads();

    int phys_id = shared_phys_id;
    int slot    = shared_slot;

    size_t block_elems = (size_t)block_size * hidden_dim;
    size_t base        = (size_t)phys_id * 2 * block_elems;

    const float* qkv_row = buf_qkv + (size_t)s * 3 * hidden_dim;

    const float* k_src = qkv_row + hidden_dim;
    const float* v_src = qkv_row + 2 * hidden_dim;

    float* k_dst = kv_pool + base + (size_t)slot * hidden_dim;
    float* v_dst = kv_pool + base + block_elems + (size_t)slot * hidden_dim;

    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        k_dst[d] = k_src[d];
        v_dst[d] = v_src[d];
    }
}

/*
 * thread 하나가 float2 단위로 K/V를 복사한다.
 */
__global__ void append_qkv_to_paged_kv_vec2_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ kv_pool,
    const int* __restrict__ block_table,
    int start_pos,
    int seq_len,
    int block_size,
    int hidden_dim
) {
    int s = blockIdx.x;
    if (s >= seq_len) return;

    __shared__ int shared_phys_id;
    __shared__ int shared_slot;

    if (threadIdx.x == 0) {
        int pos = start_pos + s;

        int logical_block = pos / block_size;
        int slot          = pos % block_size;
        int phys_id       = block_table[logical_block];

        shared_phys_id = phys_id;
        shared_slot    = slot;
    }

    __syncthreads();

    int phys_id = shared_phys_id;
    int slot    = shared_slot;

    int hidden_vec2 = hidden_dim / 2;

    size_t block_elems = (size_t)block_size * hidden_dim;
    size_t base        = (size_t)phys_id * 2 * block_elems;

    const float2* qkv2 = reinterpret_cast<const float2*>(
        buf_qkv + (size_t)s * 3 * hidden_dim
    );

    const float2* k_src2 = qkv2 + hidden_vec2;
    const float2* v_src2 = qkv2 + 2 * hidden_vec2;

    float2* k_dst2 = reinterpret_cast<float2*>(
        kv_pool + base + (size_t)slot * hidden_dim
    );

    float2* v_dst2 = reinterpret_cast<float2*>(
        kv_pool + base + block_elems + (size_t)slot * hidden_dim
    );

    for (int i = threadIdx.x; i < hidden_vec2; i += blockDim.x) {
        k_dst2[i] = k_src2[i];
        v_dst2[i] = v_src2[i];
    }
}

/*
 * thread 하나가 float4 단위로 K/V를 복사한다.
 */
__global__ void append_qkv_to_paged_kv_vec4_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ kv_pool,
    const int* __restrict__ block_table,
    int start_pos,
    int seq_len,
    int block_size,
    int hidden_dim
) {
    int s = blockIdx.x;
    if (s >= seq_len) return;

    __shared__ int shared_phys_id;
    __shared__ int shared_slot;

    if (threadIdx.x == 0) {
        int pos = start_pos + s;

        int logical_block = pos / block_size;
        int slot          = pos % block_size;
        int phys_id       = block_table[logical_block];

        shared_phys_id = phys_id;
        shared_slot    = slot;
    }

    __syncthreads();

    int phys_id = shared_phys_id;
    int slot    = shared_slot;

    int hidden_vec4 = hidden_dim / 4;

    size_t block_elems = (size_t)block_size * hidden_dim;
    size_t base        = (size_t)phys_id * 2 * block_elems;

    const float4* qkv4 = reinterpret_cast<const float4*>(
        buf_qkv + (size_t)s * 3 * hidden_dim
    );

    const float4* k_src4 = qkv4 + hidden_vec4;
    const float4* v_src4 = qkv4 + 2 * hidden_vec4;

    float4* k_dst4 = reinterpret_cast<float4*>(
        kv_pool + base + (size_t)slot * hidden_dim
    );

    float4* v_dst4 = reinterpret_cast<float4*>(
        kv_pool + base + block_elems + (size_t)slot * hidden_dim
    );

    for (int i = threadIdx.x; i < hidden_vec4; i += blockDim.x) {
        k_dst4[i] = k_src4[i];
        v_dst4[i] = v_src4[i];
    }
}

/*
 * vec8은 CUDA built-in float8이 없으므로
 * thread 하나가 float4 두 개를 복사한다.
 */
__global__ void append_qkv_to_paged_kv_vec8_kernel(
    const float* __restrict__ buf_qkv,
    float* __restrict__ kv_pool,
    const int* __restrict__ block_table,
    int start_pos,
    int seq_len,
    int block_size,
    int hidden_dim
) {
    int s = blockIdx.x;
    if (s >= seq_len) return;

    __shared__ int shared_phys_id;
    __shared__ int shared_slot;

    if (threadIdx.x == 0) {
        int pos = start_pos + s;

        int logical_block = pos / block_size;
        int slot          = pos % block_size;
        int phys_id       = block_table[logical_block];

        shared_phys_id = phys_id;
        shared_slot    = slot;
    }

    __syncthreads();

    int phys_id = shared_phys_id;
    int slot    = shared_slot;

    int hidden_vec4 = hidden_dim / 4;
    int hidden_vec8 = hidden_dim / 8;

    size_t block_elems = (size_t)block_size * hidden_dim;
    size_t base        = (size_t)phys_id * 2 * block_elems;

    const float4* qkv4 = reinterpret_cast<const float4*>(
        buf_qkv + (size_t)s * 3 * hidden_dim
    );

    const float4* k_src4 = qkv4 + hidden_vec4;
    const float4* v_src4 = qkv4 + 2 * hidden_vec4;

    float4* k_dst4 = reinterpret_cast<float4*>(
        kv_pool + base + (size_t)slot * hidden_dim
    );

    float4* v_dst4 = reinterpret_cast<float4*>(
        kv_pool + base + block_elems + (size_t)slot * hidden_dim
    );

    for (int i = threadIdx.x; i < hidden_vec8; i += blockDim.x) {
        int j = i * 2;

        k_dst4[j + 0] = k_src4[j + 0];
        k_dst4[j + 1] = k_src4[j + 1];

        v_dst4[j + 0] = v_src4[j + 0];
        v_dst4[j + 1] = v_src4[j + 1];
    }
}

static void launch_append_qkv_to_paged_kv_kernel(
    const float* buf_qkv,
    float* kv_pool,
    const int* d_block_table,
    int start_pos,
    int seq_len,
    int block_size,
    int hidden_dim
) {
    if (seq_len <= 0) return;

    constexpr int THREADS = 256;

    dim3 grid(seq_len);
    dim3 block(THREADS);

#if KV_APPEND_VEC_WIDTH == 8
    if (hidden_dim % 8 == 0) {
        append_qkv_to_paged_kv_vec8_kernel<<<grid, block>>>(
            buf_qkv,
            kv_pool,
            d_block_table,
            start_pos,
            seq_len,
            block_size,
            hidden_dim
        );
    } else {
        append_qkv_to_paged_kv_vec1_kernel<<<grid, block>>>(
            buf_qkv,
            kv_pool,
            d_block_table,
            start_pos,
            seq_len,
            block_size,
            hidden_dim
        );
    }
#elif KV_APPEND_VEC_WIDTH == 4
    if (hidden_dim % 4 == 0) {
        append_qkv_to_paged_kv_vec4_kernel<<<grid, block>>>(
            buf_qkv,
            kv_pool,
            d_block_table,
            start_pos,
            seq_len,
            block_size,
            hidden_dim
        );
    } else {
        append_qkv_to_paged_kv_vec1_kernel<<<grid, block>>>(
            buf_qkv,
            kv_pool,
            d_block_table,
            start_pos,
            seq_len,
            block_size,
            hidden_dim
        );
    }
#elif KV_APPEND_VEC_WIDTH == 2
    if (hidden_dim % 2 == 0) {
        append_qkv_to_paged_kv_vec2_kernel<<<grid, block>>>(
            buf_qkv,
            kv_pool,
            d_block_table,
            start_pos,
            seq_len,
            block_size,
            hidden_dim
        );
    } else {
        append_qkv_to_paged_kv_vec1_kernel<<<grid, block>>>(
            buf_qkv,
            kv_pool,
            d_block_table,
            start_pos,
            seq_len,
            block_size,
            hidden_dim
        );
    }
#else
    append_qkv_to_paged_kv_vec1_kernel<<<grid, block>>>(
        buf_qkv,
        kv_pool,
        d_block_table,
        start_pos,
        seq_len,
        block_size,
        hidden_dim
    );
#endif

    CUDA_CHECK(cudaGetLastError());
}

/*
 * ============================================================
 * PagedKVCache methods
 * ============================================================
 */

PagedKVCache::~PagedKVCache() {
    if (d_block_table) {
        cudaFree(d_block_table);
        d_block_table = nullptr;
    }

    d_block_table_capacity = 0;
}

void PagedKVCache::ensure_device_block_table_capacity(int needed_blocks) {
    if (needed_blocks <= d_block_table_capacity) return;

    int new_capacity = std::max(needed_blocks, std::max(4, d_block_table_capacity * 2));

    int* new_table = nullptr;
    CUDA_CHECK(cudaMalloc(&new_table, (size_t)new_capacity * sizeof(int)));

    if (d_block_table && d_block_table_capacity > 0) {
        CUDA_CHECK(cudaMemcpy(
            new_table,
            d_block_table,
            (size_t)d_block_table_capacity * sizeof(int),
            cudaMemcpyDeviceToDevice
        ));

        CUDA_CHECK(cudaFree(d_block_table));
    }

    d_block_table = new_table;
    d_block_table_capacity = new_capacity;
}

void PagedKVCache::upload_new_block_ids_to_device(
    int old_blocks,
    const std::vector<int>& new_phys_ids
) {
    if (new_phys_ids.empty()) return;

    CUDA_CHECK(cudaMemcpy(
        d_block_table + old_blocks,
        new_phys_ids.data(),
        (size_t)new_phys_ids.size() * sizeof(int),
        cudaMemcpyHostToDevice
    ));
}

void PagedKVCache::update_filled_counts() {
    for (auto& entry : block_table) {
        int block_start = entry.logical_idx * block_size;
        int filled = num_tokens - block_start;

        if (filled < 0) {
            filled = 0;
        } else if (filled > block_size) {
            filled = block_size;
        }

        entry.filled = filled;
    }
}

void PagedKVCache::reserve_blocks_for_tokens(int total_tokens) {
    if (total_tokens <= 0) return;

    int needed_blocks = (total_tokens + block_size - 1) / block_size;
    int old_blocks = (int)block_table.size();

    if (old_blocks >= needed_blocks) {
        ensure_device_block_table_capacity(needed_blocks);
        return;
    }

    nvtxRangePushA("reserve_kv_blocks");

    ensure_device_block_table_capacity(needed_blocks);

    std::vector<int> new_phys_ids;
    new_phys_ids.reserve(needed_blocks - old_blocks);

    while ((int)block_table.size() < needed_blocks) {
        int phys_id     = BlockAllocator::getInstance().allocate();
        int logical_idx = (int)block_table.size();

        block_table.push_back({
            logical_idx,
            phys_id,
            0
        });

        new_phys_ids.push_back(phys_id);
    }

    upload_new_block_ids_to_device(old_blocks, new_phys_ids);

    nvtxRangePop();
}

void PagedKVCache::append_token_kv(float* new_k, float* new_v) {
    if (block_table.empty() || block_table.back().filled >= block_size) {
        int old_blocks = (int)block_table.size();

        ensure_device_block_table_capacity(old_blocks + 1);

        int phys_id     = BlockAllocator::getInstance().allocate();
        int logical_idx = (int)block_table.size();

        block_table.push_back({
            logical_idx,
            phys_id,
            0
        });

        std::vector<int> new_phys_ids;
        new_phys_ids.push_back(phys_id);

        upload_new_block_ids_to_device(old_blocks, new_phys_ids);
    }

    BlockTableEntry& entry = block_table.back();
    PhysicalBlock&   block = BlockAllocator::getInstance().get_block(entry.phys_block_id);

    size_t sz = (size_t)hidden_dim * sizeof(float);

    CUDA_CHECK(cudaMemcpy(
        block.get_key_at(entry.filled),
        new_k,
        sz,
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        block.get_value_at(entry.filled),
        new_v,
        sz,
        cudaMemcpyDeviceToDevice
    ));

    entry.filled++;
    num_tokens++;
}

void PagedKVCache::append_token_kv_batch(float* new_k, float* new_v, int n) {
    for (int i = 0; i < n; i++) {
        append_token_kv(
            new_k + (size_t)i * hidden_dim,
            new_v + (size_t)i * hidden_dim
        );
    }
}

void PagedKVCache::append_qkv_from_interleaved(
    const float* buf_qkv,
    int seq_len
) {
    if (seq_len <= 0) return;

    int end_pos = num_tokens + seq_len;

    /*
     * 기존 API는 안전성을 위해 reserve 후 no_alloc append.
     * 이 함수도 더 이상 매번 cudaMalloc/cudaFree하지 않는다.
     */
    reserve_blocks_for_tokens(end_pos);
    append_qkv_from_interleaved_no_alloc(buf_qkv, seq_len);
}

void PagedKVCache::append_qkv_from_interleaved_no_alloc(
    const float* buf_qkv,
    int seq_len
) {
    if (seq_len <= 0) return;

    int start_pos = num_tokens;
    int end_pos   = num_tokens + seq_len;

    int needed_blocks = (end_pos + block_size - 1) / block_size;

    if ((int)block_table.size() < needed_blocks) {
        fprintf(stderr,
                "PagedKVCache::append_qkv_from_interleaved_no_alloc: "
                "not enough reserved blocks. have=%zu needed=%d\n",
                block_table.size(),
                needed_blocks);
        exit(1);
    }

    if (d_block_table == nullptr || d_block_table_capacity < needed_blocks) {
        fprintf(stderr,
                "PagedKVCache::append_qkv_from_interleaved_no_alloc: "
                "device block table not ready. capacity=%d needed=%d\n",
                d_block_table_capacity,
                needed_blocks);
        exit(1);
    }

    float* kv_pool = BlockAllocator::getInstance().get_pool();

    /*
     * 여기에는 cudaMalloc/cudaMemcpy/cudaFree가 없어야 한다.
     * 실제 K/V 이동 kernel만 실행한다.
     */
    nvtxRangePushA("kv move");
    launch_append_qkv_to_paged_kv_kernel(
        buf_qkv,
        kv_pool,
        d_block_table,
        start_pos,
        seq_len,
        block_size,
        hidden_dim
    );
    nvtxRangePop();

    num_tokens = end_pos;
    update_filled_counts();
}

PagedKVCache PagedKVCache::fork() {
    PagedKVCache new_cache(block_size, hidden_dim, /*seq_id=*/-1);

    int needed_blocks = (int)block_table.size();
    new_cache.ensure_device_block_table_capacity(needed_blocks);

    std::vector<int> phys_ids;
    phys_ids.reserve(needed_blocks);

    for (const BlockTableEntry& entry : block_table) {
        BlockAllocator::getInstance().inc_ref(entry.phys_block_id);
        new_cache.block_table.push_back(entry);
        phys_ids.push_back(entry.phys_block_id);
    }

    if (!phys_ids.empty()) {
        CUDA_CHECK(cudaMemcpy(
            new_cache.d_block_table,
            phys_ids.data(),
            (size_t)phys_ids.size() * sizeof(int),
            cudaMemcpyHostToDevice
        ));
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
                last.phys_block_id,
                last.filled
            );

            BlockAllocator::getInstance().dec_ref(last.phys_block_id);

            last.phys_block_id = new_id;

            /*
             * logical block -> physical block mapping이 바뀌었으므로
             * device block table도 해당 entry만 갱신한다.
             */
            CUDA_CHECK(cudaMemcpy(
                d_block_table + last.logical_idx,
                &new_id,
                sizeof(int),
                cudaMemcpyHostToDevice
            ));
        }
    }

    append_token_kv(key_vec, value_vec);
}

void PagedKVCache::free_all() {
    for (const BlockTableEntry& entry : block_table) {
        BlockAllocator::getInstance().free(entry.phys_block_id);
    }

    block_table.clear();
    num_tokens = 0;

    /*
     * d_block_table은 여기서 cudaFree하지 않는다.
     * 같은 PagedKVCache 객체를 재사용할 수 있으므로 capacity는 유지한다.
     * 완전 해제는 destructor에서 한다.
     */
}

void PagedKVCache::append_block_table_entry(const BlockTableEntry& bte) {
    int old_blocks = (int)block_table.size();

    ensure_device_block_table_capacity(old_blocks + 1);

    block_table.push_back(bte);

    std::vector<int> new_phys_ids;
    new_phys_ids.push_back(bte.phys_block_id);

    upload_new_block_ids_to_device(old_blocks, new_phys_ids);
}
