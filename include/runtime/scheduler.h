#pragma once
#include <thread>
#include <uv.h>
#include <atomic>
#include <iostream>
#include <deque>
#include <memory>


#include "runtime/request.h"
#include "runtime/response.h"
#include "runtime/mutex_queue.h"
#include "runtime/inference_backend.h"
#include "runtime/kv_allocator.h"
namespace mini_llm::runtime {
class Scheduler {
// Server 관련 변수들
private:
    MutexQueue<std::unique_ptr<Request>>& request_queue_;
    MutexQueue<Response>& response_queue_;
    uv_async_t* response_async_ = nullptr;
    
// Scheduler 변수들
private:
    std::deque<std::unique_ptr<Request>> waiting_prefill_queue_;
    // prefill이 가능한 queue
    std::deque<std::unique_ptr<Request>> prefill_queue_;
    std::deque<std::unique_ptr<Request>> decode_queue_;
    std::deque<std::unique_ptr<Request>> finish_queue_;
    
    std::deque<std::unique_ptr<Request>> deferred_prefill_queue_;
    std::deque<std::unique_ptr<Request>> deferred_decode_queue_;
    std::deque<std::unique_ptr<Request>> cancel_queue_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    InferenceBackend& backend_;
    KvAllocator& kv_allocator_;
public:
    Scheduler(
        MutexQueue<std::unique_ptr<Request>>& request_queue,
        MutexQueue<Response>& response_queue,
        uv_async_t* response_async,
        InferenceBackend& backend,
        KvAllocator& kv_allocator
    )
    : request_queue_(request_queue),
      response_queue_(response_queue),
      response_async_(response_async),
      backend_(backend),
      kv_allocator_(kv_allocator)   
    {
    }
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void start();
    void stop();

private:
    void worker_loop();

    bool try_admit_prefill(std::unique_ptr<Request>& req);
    bool try_admit_decode(std::unique_ptr<Request>& req);

    void drain_new_requests();
    void admit_waiting_prefill();
    void retry_deferred_prefill();
    void retry_deferred_decode();
    void run_decode_batch();
    void run_prefill_batch();
    bool no_internal_work();
    void wait_and_enqueue_one_request();
};
}