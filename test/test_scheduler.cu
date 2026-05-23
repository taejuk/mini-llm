#include "test_utils.h"
#include "scheduler/scheduler.h"
#include <thread>
#include <chrono>

static const int BLOCK_SIZE   = 16;
static const int HIDDEN_DIM   = 64;
static const int TOTAL_BLOCKS = 64;

/* ── submit / schedule 기본 테스트 ──────────────────────────── */
void test_submit_schedule() {
    TEST_BEGIN("Scheduler submit & schedule");

    Scheduler sched(BLOCK_SIZE, HIDDEN_DIM);

    // 요청 2개 제출
    auto f0 = sched.submit({1, 2, 3}, 5);
    auto f1 = sched.submit({4, 5}, 3);

    // schedule 호출 → 둘 다 PREFILL로 올라와야 함
    ScheduleBatch batch = sched.schedule();

    TEST_EQ((int)batch.prefill_reqs.size(), 2);
    TEST_EQ((int)batch.decode_reqs.size(),  0);
    TEST_ASSERT(!sched.all_done());
}

/* ── PREFILL → DECODE 전환 테스트 ───────────────────────────── */
void test_prefill_to_decode() {
    TEST_BEGIN("Scheduler PREFILL → DECODE 전환");

    Scheduler sched(BLOCK_SIZE, HIDDEN_DIM);
    sched.submit({1, 2, 3}, 2);

    ScheduleBatch batch = sched.schedule();
    TEST_EQ((int)batch.prefill_reqs.size(), 1);

    // update: 첫 번째 토큰 반환 → DECODE로 전환
    sched.update(batch, {100});

    ScheduleBatch batch2 = sched.schedule();
    TEST_EQ((int)batch2.prefill_reqs.size(), 0);
    TEST_EQ((int)batch2.decode_reqs.size(),  1);
}

/* ── max_new_tokens 도달 → 완료 테스트 ─────────────────────── */
void test_completion() {
    TEST_BEGIN("Scheduler 요청 완료 (max_new_tokens)");

    Scheduler sched(BLOCK_SIZE, HIDDEN_DIM);
    auto fut = sched.submit({1, 2}, 3);  // 토큰 3개 생성

    // prefill step
    auto b0 = sched.schedule();
    sched.update(b0, {10});             // 토큰 1

    // decode step 1
    auto b1 = sched.schedule();
    sched.update(b1, {20});             // 토큰 2

    // decode step 2 → 완료
    auto b2 = sched.schedule();
    sched.update(b2, {30});             // 토큰 3 → DONE

    TEST_ASSERT(sched.all_done());

    // future에서 결과 수신
    auto result = fut.get();
    TEST_EQ((int)result.size(), 3);
    TEST_EQ(result[0], 10);
    TEST_EQ(result[1], 20);
    TEST_EQ(result[2], 30);
}

/* ── 메모리 부족 시 waiting 대기 테스트 ─────────────────────── */
void test_memory_pressure() {
    TEST_BEGIN("Scheduler 메모리 부족 → waiting 대기");

    // 블록 4개짜리 작은 풀 (block_size=16이면 최대 4*16=64토큰)
    // 각 요청이 prompt 32토큰 → 블록 2개 필요
    // 블록 4개면 요청 2개만 동시 처리 가능
    Scheduler sched(BLOCK_SIZE, HIDDEN_DIM);

    auto f0 = sched.submit(std::vector<int>(400, 1), 1);  // 블록 2개 필요
    auto f1 = sched.submit(std::vector<int>(400, 2), 1);  // 블록 2개 필요
    auto f2 = sched.submit(std::vector<int>(400, 3), 1);  // 블록 2개 필요 → 대기

    ScheduleBatch batch = sched.schedule();

    // 블록이 4개뿐이면 f2는 waiting에 남아야 함
    int running = (int)batch.prefill_reqs.size() + (int)batch.decode_reqs.size();
    TEST_ASSERT(running <= 2);
}

/* ── 멀티스레드 submit 테스트 ────────────────────────────────── */
void test_concurrent_submit() {
    TEST_BEGIN("Scheduler 멀티스레드 submit (3 producers)");

    Scheduler sched(BLOCK_SIZE, HIDDEN_DIM);
    std::vector<std::future<std::vector<int>>> futs;
    std::mutex fut_mu;

    // 3개 스레드가 동시에 submit
    std::vector<std::thread> threads;
    for (int t = 0; t < 3; t++) {
        threads.emplace_back([&, t]() {
            auto f = sched.submit({t, t+1, t+2}, 1);
            std::lock_guard<std::mutex> lk(fut_mu);
            futs.push_back(std::move(f));
        });
    }
    for (auto& th : threads) th.join();

    // 모두 schedule로 올라오는지 확인
    ScheduleBatch batch = sched.schedule();
    TEST_EQ((int)batch.prefill_reqs.size(), 3);
}

int main() {
    printf("====== Scheduler Unit Tests ======\n");

    // BlockAllocator 초기화 (scheduler가 내부적으로 사용)
    BlockAllocator::getInstance(TOTAL_BLOCKS, BLOCK_SIZE, HIDDEN_DIM);

    test_submit_schedule();
    test_prefill_to_decode();
    test_completion();
    test_memory_pressure();
    test_concurrent_submit();

    TEST_SUMMARY();
}
