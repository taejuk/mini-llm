#include "test_utils.h"
#include "vllm/allocator.cuh"
#include "vllm/kv_cache.cuh"

static const int BLOCK_SIZE  = 16;
static const int HIDDEN_DIM  = 64;   // 테스트용 작은 크기
static const int TOTAL_BLOCKS = 32;

/* ── append_token_kv 기본 테스트 ─────────────────────────────── */
void test_append() {
    TEST_BEGIN("PagedKVCache append_token_kv");

    PagedKVCache kv(BLOCK_SIZE, HIDDEN_DIM, 0);
    TEST_EQ(kv.get_num_tokens(), 0);
    TEST_EQ(kv.get_num_blocks(), 0);

    // GPU 버퍼 준비
    float* d_k; float* d_v;
    cudaMalloc(&d_k, HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_v, HIDDEN_DIM * sizeof(float));

    std::vector<float> h_k(HIDDEN_DIM, 1.f);
    std::vector<float> h_v(HIDDEN_DIM, 2.f);
    cudaMemcpy(d_k, h_k.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // 토큰 16개 추가 → 블록 1개 소모
    for (int i = 0; i < BLOCK_SIZE; i++)
        kv.append_token_kv(d_k, d_v);

    TEST_EQ(kv.get_num_tokens(), BLOCK_SIZE);
    TEST_EQ(kv.get_num_blocks(), 1);

    // 토큰 1개 더 → 블록 2개
    kv.append_token_kv(d_k, d_v);
    TEST_EQ(kv.get_num_tokens(), BLOCK_SIZE + 1);
    TEST_EQ(kv.get_num_blocks(), 2);

    // 블록 테이블의 filled 확인
    const auto& bt = kv.get_block_table();
    TEST_EQ(bt[0].filled, BLOCK_SIZE);
    TEST_EQ(bt[1].filled, 1);

    kv.free_all();
    TEST_EQ(kv.get_num_tokens(), 0);

    cudaFree(d_k); cudaFree(d_v);
}

/* ── fork (CoW) 테스트 ───────────────────────────────────────── */
void test_fork() {
    TEST_BEGIN("PagedKVCache fork (Copy-on-Write)");

    auto& alloc = BlockAllocator::getInstance();
    int free0 = alloc.get_num_free_blocks();

    PagedKVCache kv(BLOCK_SIZE, HIDDEN_DIM, 0);
    float* d_k; float* d_v;
    cudaMalloc(&d_k, HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_v, HIDDEN_DIM * sizeof(float));

    // 토큰 10개 추가 (블록 1개, 10/16 채움)
    for (int i = 0; i < 10; i++) kv.append_token_kv(d_k, d_v);
    TEST_EQ(alloc.get_num_free_blocks(), free0 - 1);

    // fork → 블록 공유 (새 블록 할당 없음)
    PagedKVCache kv2 = kv.fork();
    TEST_EQ(kv2.get_num_tokens(), 10);
    TEST_EQ(alloc.get_num_free_blocks(), free0 - 1);  // 블록 공유

    // 공유 블록 확인 (같은 phys_block_id)
    TEST_EQ(kv.get_block_table()[0].phys_block_id,
            kv2.get_block_table()[0].phys_block_id);

    kv.free_all();
    kv2.free_all();
    TEST_EQ(alloc.get_num_free_blocks(), free0);

    cudaFree(d_k); cudaFree(d_v);
}

/* ── cow_append 테스트 ───────────────────────────────────────── */
void test_cow_append() {
    TEST_BEGIN("PagedKVCache cow_append");

    auto& alloc = BlockAllocator::getInstance();
    int free0 = alloc.get_num_free_blocks();

    PagedKVCache kv(BLOCK_SIZE, HIDDEN_DIM, 0);
    float* d_k; float* d_v;
    cudaMalloc(&d_k, HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_v, HIDDEN_DIM * sizeof(float));

    for (int i = 0; i < 10; i++) kv.append_token_kv(d_k, d_v);

    PagedKVCache kv2 = kv.fork();   // 공유 상태

    // kv2에 cow_append → 공유 블록 분리됨
    kv2.cow_append(d_k, d_v);
    TEST_EQ(kv2.get_num_tokens(), 11);

    // 이제 두 kv는 다른 물리 블록을 가져야 함
    TEST_ASSERT(kv.get_block_table()[0].phys_block_id !=
                kv2.get_block_table()[0].phys_block_id);

    // 블록 2개 사용 중 (kv:1, kv2:1)
    TEST_EQ(alloc.get_num_free_blocks(), free0 - 2);

    kv.free_all();
    kv2.free_all();
    TEST_EQ(alloc.get_num_free_blocks(), free0);

    cudaFree(d_k); cudaFree(d_v);
}

int main() {
    printf("====== PagedKVCache Unit Tests ======\n");

    BlockAllocator::getInstance(TOTAL_BLOCKS, BLOCK_SIZE, HIDDEN_DIM);

    test_append();
    test_fork();
    test_cow_append();

    TEST_SUMMARY();
}
