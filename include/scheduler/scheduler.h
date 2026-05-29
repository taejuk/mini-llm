#pragma once

#include "scheduler/mpsc_queue.h"
#include "scheduler/request.h"
#include "vllm/allocator.cuh"
#include "model/gpt2_common.cuh"
#include <deque>
#include <vector>
#include <memory>

/*
 * ScheduleBatch — 한 step에서 GPU에 보낼 배치
 *
 * prefill_reqs: 이번 step에 prefill할 요청들
 * decode_reqs : 이번 step에 decode할 요청들
 * finished    : 이번 step에 완료된 요청들 (result_promise 설정 후 제거)
 */
struct ScheduleBatch {
    std::vector<Request*> prefill_reqs;
    std::vector<Request*> decode_reqs;
    std::vector<Request*> finished;
};

/*
 * Scheduler
 *
 * - 클라이언트가 submit()으로 요청 등록
 * - Scheduler 스레드가 step()마다 schedule()을 호출해 배치 결정
 * - 메모리 기준: BlockAllocator의 free block 수로 승격 여부 판단
 *
 * 스케줄 정책 (간단 버전):
 *   1) DECODE 요청 먼저 유지 (다음 토큰에 필요한 블록 예약)
 *   2) 남은 free block으로 WAITING → PREFILL 승격
 *   3) 메모리 부족 시 waiting에서 대기
 */
class Scheduler {
public:
    explicit Scheduler(int block_size, int hidden_dim);

    /* ── Producer (클라이언트 스레드) ──────────────────────────── */
    // 요청 등록 → future로 결과 수신
    std::future<std::vector<int>> submit(std::vector<int> prompt_ids,
                                         int max_new_tokens);

    /* ── Consumer (Scheduler 스레드) ───────────────────────────── */
    // 한 step의 배치 결정
    // GPU forward pass는 호출자(scheduler loop)가 직접 수행
    ScheduleBatch schedule();

    // GPU forward 결과를 각 요청에 반영
    // next_tokens[i] = decode_reqs[i] 또는 prefill_reqs[i]의 다음 토큰
    void update(const ScheduleBatch& batch,
                const std::vector<int>& next_tokens);

    bool all_done() const;

private:
    int block_size_;
    int hidden_dim_;

    MpscQueue<std::unique_ptr<Request>> incoming_;  // 새 요청 큐
    std::deque<Request*>                waiting_;   // 메모리 대기 중
    std::vector<Request*>               running_;   // 실행 중 (PREFILL or DECODE)

    // 모든 Request의 소유권 (메모리 관리)
    std::vector<std::unique_ptr<Request>> requests_;
    int next_id_ = 0;

    /* ── 내부 헬퍼 ─────────────────────────────────────────────── */

    // waiting → running 승격 시도
    // 반환: 승격된 요청 수
    int try_admit();

    // 다음 decode step에 필요한 블록 수 추정
    int blocks_needed_for_running() const;

    // 요청 하나를 prefill하는 데 필요한 블록 수
    int blocks_needed_for(const Request* req) const;
};
