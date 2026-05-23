#include "test_utils.h"
#include "scheduler/mpsc_queue.h"
#include <thread>
#include <vector>
#include <atomic>
#include <memory>

/* ── 기본 동작 테스트 ────────────────────────────────────────── */
void test_basic() {
    TEST_BEGIN("MpscQueue 기본 동작");
    MpscQueue<int> q;

    TEST_ASSERT(q.empty());
    TEST_EQ(q.size(), 0u);

    q.push(1);
    q.push(2);
    q.push(3);

    TEST_ASSERT(!q.empty());
    TEST_EQ(q.size(), 3u);

    auto v = q.try_pop();
    TEST_ASSERT(v.has_value());
    TEST_EQ(v.value(), 1);
    TEST_EQ(q.size(), 2u);

    auto v2 = q.try_pop();
    TEST_EQ(v2.value(), 2);

    auto v3 = q.try_pop();
    TEST_EQ(v3.value(), 3);

    TEST_ASSERT(q.empty());

    // 빈 큐에서 pop
    auto empty = q.try_pop();
    TEST_ASSERT(!empty.has_value());
}

/* ── drain 테스트 ────────────────────────────────────────────── */
void test_drain() {
    TEST_BEGIN("MpscQueue drain");
    MpscQueue<int> q;

    for (int i = 0; i < 5; i++) q.push(i);

    std::vector<int> out;
    q.drain(out);

    TEST_EQ(out.size(), 5u);
    TEST_ASSERT(q.empty());
    TEST_EQ(out[0], 0);
    TEST_EQ(out[4], 4);

    // 빈 큐 drain
    std::vector<int> out2;
    q.drain(out2);
    TEST_EQ(out2.size(), 0u);
}

/* ── unique_ptr (move-only) 타입 테스트 ─────────────────────── */
void test_move_only() {
    TEST_BEGIN("MpscQueue unique_ptr (move-only 타입)");
    MpscQueue<std::unique_ptr<int>> q;

    q.push(std::make_unique<int>(42));
    q.push(std::make_unique<int>(99));

    auto v = q.try_pop();
    TEST_ASSERT(v.has_value());
    TEST_EQ(*v.value(), 42);

    std::vector<std::unique_ptr<int>> out;
    q.drain(out);
    TEST_EQ(out.size(), 1u);
    TEST_EQ(*out[0], 99);
}

/* ── 멀티스레드 테스트 ───────────────────────────────────────── */
void test_multithread() {
    TEST_BEGIN("MpscQueue 멀티스레드 (4 producers)");
    MpscQueue<int> q;
    std::atomic<int> total{0};

    const int N_PRODUCERS = 4;
    const int N_ITEMS     = 1000;

    // 4개 스레드가 동시에 push
    std::vector<std::thread> producers;
    for (int t = 0; t < N_PRODUCERS; t++) {
        producers.emplace_back([&q, t]() {
            for (int i = 0; i < N_ITEMS; i++)
                q.push(t * N_ITEMS + i);
        });
    }
    for (auto& th : producers) th.join();

    TEST_EQ(q.size(), (size_t)(N_PRODUCERS * N_ITEMS));

    // Single consumer drain
    std::vector<int> out;
    q.drain(out);
    TEST_EQ(out.size(), (size_t)(N_PRODUCERS * N_ITEMS));
    TEST_ASSERT(q.empty());
}

int main() {
    printf("====== MpscQueue Unit Tests ======\n");
    test_basic();
    test_drain();
    test_move_only();
    test_multithread();
    TEST_SUMMARY();
}
