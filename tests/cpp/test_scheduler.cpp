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

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;

namespace {

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

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

} // namespace

int main() {
    test_single_request_lifecycle();
    test_two_requests_finish();

    std::cout << "[PASS] scheduler tests\n";
    return 0;
}
