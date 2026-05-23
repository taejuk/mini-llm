#include "test_utils.h"
#include "vllm/allocator.cuh"

/* ── 초기화 / 기본 할당 ──────────────────────────────────────── */
void test_basic_alloc() {
    TEST_BEGIN("BlockAllocator 기본 할당 / 해제");

    auto& alloc = BlockAllocator::getInstance();
    int total = alloc.get_total_blocks();
    int free0  = alloc.get_num_free_blocks();
    TEST_EQ(free0, total);

    int id0 = alloc.allocate();
    TEST_EQ(alloc.get_num_free_blocks(), total - 1);

    int id1 = alloc.allocate();
    TEST_EQ(alloc.get_num_free_blocks(), total - 2);

    alloc.free(id0);
    TEST_EQ(alloc.get_num_free_blocks(), total - 1);

    alloc.free(id1);
    TEST_EQ(alloc.get_num_free_blocks(), total);
}

/* ── refcount 테스트 ─────────────────────────────────────────── */
void test_refcount() {
    TEST_BEGIN("BlockAllocator refcount");

    auto& alloc = BlockAllocator::getInstance();
    int free0 = alloc.get_num_free_blocks();

    int id = alloc.allocate();          // refcount = 1
    alloc.inc_ref(id);                  // refcount = 2
    alloc.free(id);                     // refcount = 1 → 해제 안 됨
    TEST_EQ(alloc.get_num_free_blocks(), free0 - 1);

    alloc.free(id);                     // refcount = 0 → 해제
    TEST_EQ(alloc.get_num_free_blocks(), free0);
}

/* ── copy_block 테스트 ───────────────────────────────────────── */
void test_copy_block() {
    TEST_BEGIN("BlockAllocator copy_block (CoW)");

    auto& alloc = BlockAllocator::getInstance();
    int free0 = alloc.get_num_free_blocks();

    // src 블록에 데이터 쓰기
    int src_id = alloc.allocate();
    PhysicalBlock& src = alloc.get_block(src_id);

    int hidden = alloc.get_hidden_dim();
    std::vector<float> h_data(hidden, 1.23f);
    cudaMemcpy(src.get_key_at(0), h_data.data(),
               hidden * sizeof(float), cudaMemcpyHostToDevice);

    // copy_block: filled=1 슬롯 복사
    int dst_id = alloc.copy_block(src_id, 1);
    TEST_ASSERT(dst_id != src_id);
    TEST_EQ(alloc.get_num_free_blocks(), free0 - 2);

    // 복사된 데이터 확인
    PhysicalBlock& dst = alloc.get_block(dst_id);
    std::vector<float> h_out(hidden, 0.f);
    cudaMemcpy(h_out.data(), dst.get_key_at(0),
               hidden * sizeof(float), cudaMemcpyDeviceToHost);

    bool match = true;
    for (int i = 0; i < hidden; i++)
        if (h_out[i] != 1.23f) { match = false; break; }
    TEST_ASSERT(match);

    alloc.free(src_id);
    alloc.free(dst_id);
    TEST_EQ(alloc.get_num_free_blocks(), free0);
}

int main() {
    printf("====== BlockAllocator Unit Tests ======\n");

    // Singleton 초기화 (총 64블록, block_size=16, hidden=768)
    BlockAllocator::getInstance(64, 16, 768);

    test_basic_alloc();
    test_refcount();
    test_copy_block();

    TEST_SUMMARY();
}
