#include "runtime/scheduler.h"
#include "runtime/block.h"

#include <utility>
#include <vector>

namespace mini_llm::runtime {

namespace C = mini_llm::constants;

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::start() {
    running_.store(true);
    worker_ = std::thread([this] {
        worker_loop();
    });
}

void Scheduler::stop() {
    running_.store(false);
    request_queue_.close();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool Scheduler::try_admit_prefill(std::unique_ptr<Request>& req) {
    return kv_allocator_.allocate_prefill(*req);
}

bool Scheduler::try_admit_decode(std::unique_ptr<Request>& req) {
    return kv_allocator_.allocate_decode(*req);
}

void Scheduler::drain_new_requests() {
    std::unique_ptr<Request> req;
    int nums = 0;
    while (nums < C::MAX_BATCH_NUM && request_queue_.try_pop(req)) {
        waiting_prefill_queue_.push_back(std::move(req));
        nums++;
    }
}

bool Scheduler::admit_waiting_prefill() {
    int n = static_cast<int>(waiting_prefill_queue_.size());
    bool progress = false;

    for (int i = 0; i < n; i++) {
        auto req = std::move(waiting_prefill_queue_.front());
        waiting_prefill_queue_.pop_front();

        if (try_admit_prefill(req)) {
            req->state = RequestState::PrefillRunning;
            prefill_queue_.push_back(std::move(req));
            progress = true;
        } else {
            req->state = RequestState::DeferredPrefillKV;
            deferred_prefill_queue_.push_back(std::move(req));
            break;
        }
    }

    return progress;
}

bool Scheduler::retry_deferred_prefill() {
    int n = static_cast<int>(deferred_prefill_queue_.size());
    bool progress = false;

    for (int i = 0; i < n; i++) {
        auto req = std::move(deferred_prefill_queue_.front());
        deferred_prefill_queue_.pop_front();

        if (try_admit_prefill(req)) {
            req->state = RequestState::PrefillRunning;
            prefill_queue_.push_front(std::move(req));
            progress = true;
        } else {
            deferred_prefill_queue_.push_back(std::move(req));
        }
    }

    return progress;
}

bool Scheduler::retry_deferred_decode() {
    int n = static_cast<int>(deferred_decode_queue_.size());

    for (int i = 0; i < n; i++) {
        auto req = std::move(deferred_decode_queue_.front());
        deferred_decode_queue_.pop_front();
        req->state = RequestState::DecodeReady;
        decode_queue_.push_back(std::move(req));
    }

    return false;
}
// 이거를 비동기 함수로 바꿔야 한다. swap_in을 비동기로 바꿔야 한다.

bool Scheduler::retry_swap_in_requests() {
    int n = static_cast<int>(swap_out_queue_.size());
    bool progress = false;

    for (int i = 0; i < n; i++) {
        auto req = std::move(swap_out_queue_.front());
        swap_out_queue_.pop_front();
        // 
        if (kv_allocator_.swap_in(*req)) {
            req->state = RequestState::DecodeReady;
            decode_queue_.push_back(std::move(req));
            progress = true;
        } else {
            req->state = RequestState::SwappedOut;
            swap_out_queue_.push_back(std::move(req));
        }
    }

    return progress;
}

bool Scheduler::run_decode_batch() {
    bool has_resp = false;
    bool progress = false;
    int remaining = static_cast<int>(decode_queue_.size());
    std::deque<std::unique_ptr<Request>> next_decode_queue;

    while (remaining > 0) {
        std::vector<std::unique_ptr<Request>> batch;

        while (remaining > 0 && static_cast<int>(batch.size()) < C::MAX_BATCH_NUM) {
            auto req = std::move(decode_queue_.front());
            decode_queue_.pop_front();
            remaining--;

            if (try_admit_decode(req)) {
                req->state = RequestState::DecodeRunning;
                batch.push_back(std::move(req));
            } else {
                req->state = RequestState::DeferredDecodeKV;
                deferred_decode_queue_.push_back(std::move(req));
            }
        }

        if (batch.empty()) {
            continue;
        }

        std::vector<Response> responses = backend_.decode(batch);
        progress = true;

        if (responses.size() != batch.size()) {
            std::cerr << "Scheduler: decode response size mismatch. "
                      << "responses=" << responses.size()
                      << ", batch=" << batch.size() << "\n";
        }

        for (size_t i = 0; i < batch.size(); i++) {
            auto req = std::move(batch[i]);

            Response resp;
            if (i < responses.size()) {
                resp = responses[i];
            } else {
                resp = Response(req->request_id, -1, true);
            }

            for (auto& kv : req->layer_kv) {
                kv.increment_tokens(1);
            }

            req->tokens.push_back(resp.token);

            if (req->isfinish()) {
                resp.finished = true;
            }

            response_queue_.push(resp);
            has_resp = true;

            if (resp.finished) {
                kv_allocator_.free_request(*req);
                req->state = RequestState::Finished;
                finish_queue_.push_back(std::move(req));
            } else {
                req->state = RequestState::DecodeReady;
                next_decode_queue.push_back(std::move(req));
            }
        }
    }

    while (!next_decode_queue.empty()) {
        decode_queue_.push_back(std::move(next_decode_queue.front()));
        next_decode_queue.pop_front();
    }

    if (has_resp && response_async_ != nullptr) {
        uv_async_send(response_async_);
    }

    return progress;
}

bool Scheduler::run_prefill_batch() {
    bool has_resp = false;
    bool progress = false;

    while (!prefill_queue_.empty()) {
        std::vector<std::unique_ptr<Request>> batch;

        while (!prefill_queue_.empty() && static_cast<int>(batch.size()) < C::MAX_BATCH_NUM) {
            auto req = std::move(prefill_queue_.front());
            prefill_queue_.pop_front();
            req->state = RequestState::PrefillRunning;
            batch.push_back(std::move(req));
        }

        if (batch.empty()) {
            break;
        }

        std::vector<Response> responses = backend_.prefill(batch);
        progress = true;

        for (size_t i = 0; i < batch.size(); i++) {
            auto req = std::move(batch[i]);

            Response resp;
            if (i < responses.size()) {
                resp = responses[i];
            } else {
                resp = Response(req->request_id, -1, true);
            }

            for (auto& kv : req->layer_kv) {
                kv.set_num_tokens(req->prompts_len);
            }

            req->tokens.push_back(resp.token);

            if (req->isfinish()) {
                resp.finished = true;
            }

            response_queue_.push(resp);
            has_resp = true;

            if (resp.finished) {
                kv_allocator_.free_request(*req);
                req->state = RequestState::Finished;
                finish_queue_.push_back(std::move(req));
            } else {
                req->state = RequestState::DecodeReady;
                decode_queue_.push_back(std::move(req));
            }
        }
    }

    if (has_resp && response_async_ != nullptr) {
        uv_async_send(response_async_);
    }

    return progress;
}

bool Scheduler::no_internal_work() {
    return waiting_prefill_queue_.empty()
        && prefill_queue_.empty()
        && decode_queue_.empty()
        && deferred_prefill_queue_.empty()
        && deferred_decode_queue_.empty()
        && swap_out_queue_.empty()
        && cancel_queue_.empty();
}

void Scheduler::wait_and_enqueue_one_request() {
    auto req = request_queue_.wait_pop();
    if (!req) {
        running_.store(false);
        return;
    }
    waiting_prefill_queue_.push_back(std::move(req));
}

bool Scheduler::try_swap_out_victim() {
    auto try_from = [this](std::deque<std::unique_ptr<Request>>& queue) -> bool {
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if ((*it)->kv_residency != KvCacheResidency::Gpu) {
                continue;
            }
            if (kv_allocator_.is_swapped(**it)) {
                continue;
            }

            auto req = std::move(*it);
            queue.erase(it);
            RequestState old_state = req->state;

            if (!kv_allocator_.swap_out(*req)) {
                req->state = old_state;
                queue.push_back(std::move(req));
                return false;
            }

            req->state = RequestState::SwappedOut;
            swap_out_queue_.push_back(std::move(req));
            admission_paused_ = true;
            return true;
        }
        return false;
    };

    if (try_from(deferred_decode_queue_)) {
        return true;
    }

    return try_from(decode_queue_);
}

void Scheduler::worker_loop() {
    while (running_.load()) {
        bool is_progress = false;

        if (!admission_paused_) {
            drain_new_requests();
        }

        is_progress |= retry_swap_in_requests();

        retry_deferred_decode();
        is_progress |= run_decode_batch();

        is_progress |= retry_swap_in_requests();

        is_progress |= retry_deferred_prefill();
        is_progress |= admit_waiting_prefill();
        is_progress |= run_prefill_batch();

        is_progress |= retry_swap_in_requests();

        if (!is_progress && !no_internal_work()) {
            is_progress |= try_swap_out_victim();
        }

        if (admission_paused_ && no_internal_work()) {
            admission_paused_ = false;
        }

        if (no_internal_work()) {
            wait_and_enqueue_one_request();
        } else if (!is_progress) {
            std::this_thread::yield();
        }
    }
}

} // namespace mini_llm::runtime
