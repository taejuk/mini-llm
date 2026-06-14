#include "runtime/scheduler.h"
#include "runtime/block.h"

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
    if(worker_.joinable()) worker_.join();
    
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
    while(nums < C::MAX_BATCH_NUM && request_queue_.try_pop(req)) {
        waiting_prefill_queue_.push_back(std::move(req));
        nums++;
    }
    return;
}

void Scheduler::admit_waiting_prefill() {
    int n = waiting_prefill_queue_.size();

    for (int i = 0; i < n; i++) {
        auto req = std::move(waiting_prefill_queue_.front());
        waiting_prefill_queue_.pop_front();

        if (try_admit_prefill(req)) {
            req->state = RequestState::PrefillRunning;
            prefill_queue_.push_back(std::move(req));
        } else {
            req->state = RequestState::DeferredPrefillKV;
            deferred_prefill_queue_.push_back(std::move(req));
            break; // FCFS 유지
        }
    }
}

void Scheduler::retry_deferred_prefill() {
    // prefill이 가능한 queue에 넣는다.
    // 이 때 우선순위로 넣어야 하니깐 queue의 앞에 넣는다.
    int n = deferred_prefill_queue_.size();
    for(int i = 0; i < n; i++) {
        auto req = std::move(deferred_prefill_queue_.front());
        deferred_prefill_queue_.pop_front();

        if (try_admit_prefill(req)) prefill_queue_.push_front(std::move(req));
        else {
            deferred_prefill_queue_.push_back(std::move(req));
        } 
    }
}

void Scheduler::retry_deferred_decode() {
    int n = deferred_decode_queue_.size();
    for(int i = 0; i < n; i++) {
        auto req = std::move(deferred_decode_queue_.front());
        deferred_decode_queue_.pop_front();
        if(try_admit_decode(req)) decode_queue_.push_front(std::move(req));
        else deferred_decode_queue_.push_back(std::move(req));
    }
}

void Scheduler::run_decode_batch() {
    bool has_resp = false;

    int remaining = static_cast<int>(decode_queue_.size());
    std::deque<std::unique_ptr<Request>> next_decode_queue;

    while (remaining > 0) {
        std::vector<std::unique_ptr<Request>> batch;

        while (remaining > 0 &&
               static_cast<int>(batch.size()) < C::MAX_BATCH_NUM) {
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
                // decode가 아직 미완성이라면 여기로 들어올 수 있음
                resp = Response(req->request_id, -1, true);
            }

            // decode는 마지막 input token의 KV를 cache에 저장한 뒤
            // 다음 token을 response로 만든다고 보면 됨
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
                for (auto& kv : req->layer_kv) {
                    block_manager_.free(kv.block_table_);
                    kv.reset();
                }

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
}



void Scheduler::run_prefill_batch() {
    bool has_resp = false;

    while (!prefill_queue_.empty()) {
        std::vector<std::unique_ptr<Request>> batch;

        while (!prefill_queue_.empty() &&
               static_cast<int>(batch.size()) < C::MAX_BATCH_NUM) {
            auto req = std::move(prefill_queue_.front());
            prefill_queue_.pop_front();

            req->state = RequestState::PrefillRunning;
            batch.push_back(std::move(req));
        }

        if (batch.empty()) {
            break;
        }

        std::vector<Response> responses = backend_.prefill(batch);

        for (size_t i = 0; i < batch.size(); i++) {
            auto req = std::move(batch[i]);

            Response resp;
            if (i < responses.size()) {
                resp = responses[i];
            } else {
                resp = Response(req->request_id, -1, true);
            }

            // prefill은 prompt KV만 cache에 들어간 상태
            for (auto& kv : req->layer_kv) {
                kv.set_num_tokens(req->prompts_len);
            }

            // model이 뽑은 첫 output token은 request tokens에 추가
            req->tokens.push_back(resp.token);

            // max_new_tokens 기준 finish는 scheduler가 최종 판단
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
}


bool Scheduler::no_internal_work() {
    return waiting_prefill_queue_.empty()
        && prefill_queue_.empty()
        && decode_queue_.empty()
        && deferred_prefill_queue_.empty()
        && deferred_decode_queue_.empty()
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

// 여기에 로직이 담겨져 있어야 한다.
void Scheduler::worker_loop() {
    // 여기서 model을 가지고 와야 한다.
    while (running_.load()) {
        drain_new_requests();

        retry_deferred_decode();
        retry_deferred_prefill();

        admit_waiting_prefill();

        run_decode_batch();
        run_prefill_batch();

        if(no_internal_work()) wait_and_enqueue_one_request();    
        
    }
    // 여기서 호출하면 된다.
}
}