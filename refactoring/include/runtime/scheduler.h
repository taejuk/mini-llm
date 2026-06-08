#pragma once
#include <thread>
#include <uv.h>
#include <atomic>
#include <iostream>
#include "runtime/request.h"
#include "runtime/response.h"
#include "runtime/mutex_queue.h"

namespace mini_llm::runtime {
class Scheduler {
// Server 관련 변수들
private:
    MutexQueue<std::unique_ptr<Request>>& request_queue_;
    MutexQueue<Response>& response_queue_;
    uv_async_t* response_async_ = nullptr;
// Scheduler 변수들
private:
    MutexQueue<std::unique_ptr<Request>> waiting_prefill_queue_;
    MutexQueue<std::unique_ptr<Request>> prefill_queue_;
    MutexQueue<std::unique_ptr<Request>> decode_queue_;
    MutexQueue<std::unique_ptr<Request>> finish_queue_;
    MutexQueue<std::unique_ptr<Request>> cancel_queue_;
    std::atomic<bool> running_{false};
    std::thread worker_;
public:
    Scheduler(MutexQueue<std::unique_ptr<Request>>& request_queue, MutexQueue<Response>& response_queue, uv_async_t* response_async)
        : request_queue_(request_queue),
          response_queue_(response_queue),
          response_async_(response_async)
        {}
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void start();
    void stop();

private:
    void worker_loop();
};
}