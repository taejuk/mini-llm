#include "constants.h"
#include "runtime/mock_backend.h"
#include "runtime/mock_kv_allocator.h"
#include "runtime/mutex_queue.h"
#include "runtime/request.h"
#include "runtime/response.h"
#include "runtime/scheduler.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>
namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;

namespace {

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

class PressureKvAllocator final : public Rt::KvAllocator {
public:
    std::atomic<int> prefill_calls{0};
    std::atomic<int> decode_calls{0};
    std::atomic<int> free_calls{0};
    std::atomic<int> swap_out_calls{0};
    std::atomic<int> swap_in_calls{0};

    std::atomic<bool> fail_decode{false};
    std::atomic<bool> allow_swap_in{false};

    bool allocate_prefill(Rt::Request& req) override {
        int call_id = ++prefill_calls;

        int blocks_per_layer =
            (req.prompts_len + C::DEFAULT_KV_BLOCK_SIZE - 1) /
            C::DEFAULT_KV_BLOCK_SIZE;

        for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
            for (int b = 0; b < blocks_per_layer; ++b) {
                int fake_block_id =
                    100000 + call_id * 1000 + layer * 100 + b;
                req.layer_kv[layer].append_block(fake_block_id);
            }
        }

        req.kv_residency = Rt::KvCacheResidency::Gpu;
        return true;
    }

    bool allocate_decode(Rt::Request& req) override {
        decode_calls++;

        if (fail_decode.load()) {
            return false;
        }

        req.kv_residency = Rt::KvCacheResidency::Gpu;
        return true;
    }

    void free_request(Rt::Request& req) override {
        free_calls++;

        for (auto& kv : req.layer_kv) {
            kv.reset();
        }

        req.kv_residency = Rt::KvCacheResidency::None;
    }

    bool swap_out(Rt::Request& req) override {
        swap_out_calls++;

        req.kv_residency = Rt::KvCacheResidency::Cpu;
        req.state = Rt::RequestState::SwappedOut;
        return true;
    }

    bool swap_in(Rt::Request& req) override {
        swap_in_calls++;

        if (!allow_swap_in.load()) {
            return false;
        }

        req.kv_residency = Rt::KvCacheResidency::Gpu;
        req.state = Rt::RequestState::DecodeReady;
        return true;
    }

    bool is_swapped(const Rt::Request& req) const override {
        return req.kv_residency == Rt::KvCacheResidency::Cpu;
    }
};

std::unordered_map<uint64_t, std::vector<Rt::Response>> collect_until_done(
    Rt::MutexQueue<Rt::Response>& response_queue,
    const std::vector<uint64_t>& request_ids,
    int timeout_ms
) {
    std::unordered_map<uint64_t, std::vector<Rt::Response>> by_request;
    std::unordered_map<uint64_t, bool> finished;

    for (uint64_t id : request_ids) {
        finished[id] = false;
    }

    auto start = std::chrono::steady_clock::now();

    while (true) {
        Rt::Response resp;

        while (response_queue.try_pop(resp)) {
            if (finished.find(resp.request_id) == finished.end()) {
                continue;
            }

            by_request[resp.request_id].push_back(resp);

            if (resp.finished) {
                finished[resp.request_id] = true;
            }
        }

        bool all_finished = true;
        for (const auto& [_, done] : finished) {
            if (!done) {
                all_finished = false;
                break;
            }
        }

        if (all_finished) {
            return by_request;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start
        ).count();

        if (elapsed > timeout_ms) {
            std::cerr << "timeout while waiting scheduler responses\n";
            return by_request;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void test_single_request_lifecycle() {
    Rt::MutexQueue<std::unique_ptr<Rt::Request>> request_queue;
    Rt::MutexQueue<Rt::Response> response_queue;

    Rt::MockBackend backend;
    Rt::MockKvAllocator kv_allocator;

    Rt::Scheduler scheduler(
        request_queue,
        response_queue,
        nullptr,
        backend,
        kv_allocator
    );

    scheduler.start();

    constexpr uint64_t REQ_ID = 42;
    constexpr int MAX_NEW_TOKENS = 4;

    auto req = std::make_unique<Rt::Request>(
        REQ_ID,
        std::vector<int>{15496, 11, 616, 1438, 318},
        MAX_NEW_TOKENS
    );

    request_queue.push(std::move(req));

    auto responses_by_id = collect_until_done(
        response_queue,
        {REQ_ID},
        3000
    );

    request_queue.close();
    scheduler.stop();

    const auto& responses = responses_by_id[REQ_ID];

    check(
        static_cast<int>(responses.size()) == MAX_NEW_TOKENS,
        "single request should emit max_new_tokens responses"
    );

    check(responses[0].request_id == REQ_ID, "prefill request id mismatch");
    check(responses[0].token == 1000, "prefill token mismatch");
    check(!responses[0].finished, "prefill should not finish");

    check(responses[1].token == 1001, "first decode token mismatch");
    check(!responses[1].finished, "first decode should not finish");

    check(responses[2].token == 1002, "second decode token mismatch");
    check(!responses[2].finished, "second decode should not finish");

    check(
        responses[3].token == C::GPT2_EOS_TOKEN_ID,
        "final token should be EOS"
    );
    check(responses[3].finished, "final response should finish");

    std::cout << "[PASS] scheduler single request lifecycle\n";
}

void test_two_requests_finish() {
    Rt::MutexQueue<std::unique_ptr<Rt::Request>> request_queue;
    Rt::MutexQueue<Rt::Response> response_queue;

    Rt::MockBackend backend;
    Rt::MockKvAllocator kv_allocator;

    Rt::Scheduler scheduler(
        request_queue,
        response_queue,
        nullptr,
        backend,
        kv_allocator
    );

    scheduler.start();

    constexpr uint64_t REQ0 = 100;
    constexpr uint64_t REQ1 = 101;
    constexpr int MAX_NEW_TOKENS = 3;

    request_queue.push(std::make_unique<Rt::Request>(
        REQ0,
        std::vector<int>{15496, 11, 616},
        MAX_NEW_TOKENS
    ));

    request_queue.push(std::make_unique<Rt::Request>(
        REQ1,
        std::vector<int>{15496, 11, 616, 1438, 318},
        MAX_NEW_TOKENS
    ));

    auto responses_by_id = collect_until_done(
        response_queue,
        {REQ0, REQ1},
        3000
    );

    request_queue.close();
    scheduler.stop();

    const auto& r0 = responses_by_id[REQ0];
    const auto& r1 = responses_by_id[REQ1];

    check(
        static_cast<int>(r0.size()) == MAX_NEW_TOKENS,
        "REQ0 response count mismatch"
    );
    check(
        static_cast<int>(r1.size()) == MAX_NEW_TOKENS,
        "REQ1 response count mismatch"
    );

    check(r0.front().token == 1000, "REQ0 prefill token mismatch");
    check(r1.front().token == 1000, "REQ1 prefill token mismatch");

    check(r0.back().token == C::GPT2_EOS_TOKEN_ID, "REQ0 final token mismatch");
    check(r1.back().token == C::GPT2_EOS_TOKEN_ID, "REQ1 final token mismatch");

    check(r0.back().finished, "REQ0 should finish");
    check(r1.back().finished, "REQ1 should finish");

    std::cout << "[PASS] scheduler two requests finish\n";
}

void test_scheduler_swaps_on_global_stall() {
    Rt::MutexQueue<std::unique_ptr<Rt::Request>> request_queue;
    Rt::MutexQueue<Rt::Response> response_queue;

    Rt::MockBackend backend;
    PressureKvAllocator kv_allocator;

    kv_allocator.fail_decode.store(true);
    kv_allocator.allow_swap_in.store(false);

    Rt::Scheduler scheduler(
        request_queue,
        response_queue,
        nullptr,
        backend,
        kv_allocator
    );

    scheduler.start();

    constexpr uint64_t REQ_ID = 200;
    constexpr int MAX_NEW_TOKENS = 4;

    request_queue.push(std::make_unique<Rt::Request>(
        REQ_ID,
        std::vector<int>{15496, 11, 616, 1438, 318},
        MAX_NEW_TOKENS
    ));

    bool got_prefill = false;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        Rt::Response resp;

        while (response_queue.try_pop(resp)) {
            if (resp.request_id == REQ_ID && resp.token == 1000) {
                got_prefill = true;
            }
        }

        if (got_prefill && kv_allocator.swap_out_calls.load() > 0) {
            break;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start
        ).count();

        if (elapsed > 1000) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    request_queue.close();
    scheduler.stop();

    check(got_prefill, "stall test should receive prefill response");
    check(
        kv_allocator.swap_out_calls.load() > 0,
        "scheduler should call swap_out on global stall"
    );

    std::cout << "[PASS] scheduler swaps on global stall\n";
}

void test_scheduler_swap_in_and_finish() {
    Rt::MutexQueue<std::unique_ptr<Rt::Request>> request_queue;
    Rt::MutexQueue<Rt::Response> response_queue;

    Rt::MockBackend backend;
    PressureKvAllocator kv_allocator;

    kv_allocator.fail_decode.store(true);
    kv_allocator.allow_swap_in.store(false);

    Rt::Scheduler scheduler(
        request_queue,
        response_queue,
        nullptr,
        backend,
        kv_allocator
    );

    scheduler.start();

    constexpr uint64_t REQ_ID = 201;
    constexpr int MAX_NEW_TOKENS = 4;

    request_queue.push(std::make_unique<Rt::Request>(
        REQ_ID,
        std::vector<int>{15496, 11, 616, 1438, 318},
        MAX_NEW_TOKENS
    ));

    std::unordered_map<uint64_t, std::vector<Rt::Response>> by_request;

    bool released_pressure = false;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        Rt::Response resp;

        while (response_queue.try_pop(resp)) {
            if (resp.request_id != REQ_ID) {
                continue;
            }

            by_request[REQ_ID].push_back(resp);

            if (resp.finished) {
                request_queue.close();
                scheduler.stop();

                const auto& responses = by_request[REQ_ID];

                check(
                    static_cast<int>(responses.size()) == MAX_NEW_TOKENS,
                    "swap-in finish response count mismatch"
                );

                check(
                    responses.front().token == 1000,
                    "swap-in finish should start with prefill token"
                );

                check(
                    responses.back().token == C::GPT2_EOS_TOKEN_ID,
                    "swap-in finish should end with EOS"
                );

                check(
                    responses.back().finished,
                    "swap-in finish final response should be finished"
                );

                check(
                    kv_allocator.swap_out_calls.load() > 0,
                    "swap-in finish should call swap_out"
                );

                check(
                    kv_allocator.swap_in_calls.load() > 0,
                    "swap-in finish should call swap_in"
                );

                std::cout << "[PASS] scheduler swap-in and finish\n";
                return;
            }
        }

        if (kv_allocator.swap_out_calls.load() > 0 && !released_pressure) {
            kv_allocator.fail_decode.store(false);
            kv_allocator.allow_swap_in.store(true);
            released_pressure = true;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start
        ).count();

        if (elapsed > 3000) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    request_queue.close();
    scheduler.stop();

    check(false, "scheduler swap-in and finish timed out");
}


} // namespace

int main() {
    test_single_request_lifecycle();
    test_two_requests_finish();
    test_scheduler_swaps_on_global_stall();
    test_scheduler_swap_in_and_finish();

    std::cout << "[PASS] scheduler tests\n";
    return 0;
}
