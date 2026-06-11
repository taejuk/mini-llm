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
    worker_.detach();
}

void Scheduler::stop() {
    running_.store(false);
    //if(worker_.joinable()) worker_.join();
    
}

bool Scheduler::try_admit_prefill(std::unique_ptr<Request>& req) {
    int need_blocks_per_layer = (req->tokens.size() + C::DEFAULT_KV_BLOCK_SIZE - 1) / C::DEFAULT_KV_BLOCK_SIZE;
    int total_need = need_blocks_per_layer * C::GPT2_N_LAYERS;

    if (!block_manager_.can_allocate(total_need)) {
        return false;
    }

    for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        for (int i = 0; i < need_blocks_per_layer; i++) {
            int block_id = block_manager_.allocate_one();
            req->layer_kv[layer].append_block(block_id);
        }
    }

    return true;
}

bool Scheduler::try_admit_decode(std::unique_ptr<Request>& req) {
    // 추가한다는 것은 
    int need_blocks_per_layer = (req->tokens.size() %  C::DEFAULT_KV_BLOCK_SIZE) == 0 ? 1 : 0;
    int total_need = need_blocks_per_layer * C::GPT2_N_LAYERS;
    if(!block_manager_.can_allocate(total_need)) return false;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; layer++) {
        for (int i = 0; i < need_blocks_per_layer; i++) {
            int block_id = block_manager_.allocate_one();
            req->layer_kv[layer].append_block(block_id);
        }
    }

    return true;
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
    while (!decode_queue_.empty()) {
        std::vector<std::unique_ptr<Request>> batch;

        int n = static_cast<int>(decode_queue_.size());

        for (int i = 0;
             i < n && static_cast<int>(batch.size()) < C::MAX_BATCH_NUM;
             i++) {
            auto req = std::move(decode_queue_.front());
            decode_queue_.pop_front();

            if (try_admit_decode(req)) {
                req->state = RequestState::DecodeRunning;
                batch.push_back(std::move(req));
            } else {
                req->state = RequestState::DeferredDecodeKV;
                deferred_decode_queue_.push_back(std::move(req));
            }
        }

        if (batch.empty()) {
            break;
        }

        std::vector<Response> responses = model_.decode(batch);

        for (size_t i = 0; i < batch.size(); i++) {
            auto req = std::move(batch[i]);

            Response resp;
            if (i < responses.size()) {
                resp = responses[i];
            } else {
                resp = Response(req->request_id, -1, true);
            }

            response_queue_.push(resp);
            has_resp = true;
            

            if (resp.finished ||
                static_cast<int>(req->tokens.size()) >=
                    req->max_new_tokens + static_cast<int>(req->layer_kv[0].num_tokens_)) {
                for (auto& kv : req->layer_kv) {
                    block_manager_.free(kv.block_table_);
                    kv.reset();
                }

                req->state = RequestState::Finished;
                finish_queue_.push_back(std::move(req));
            } else {
                req->tokens.push_back(resp.token);

                for (auto& kv : req->layer_kv) {
                    kv.increment_tokens(1);
                }

                req->state = RequestState::DecodeReady;
                decode_queue_.push_back(std::move(req));
            }
        }
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

        std::vector<Response> responses = model_.prefill(batch);

        for (size_t i = 0; i < batch.size(); i++) {
            auto req = std::move(batch[i]);

            Response resp;
            if (i < responses.size()) {
                resp = responses[i];
            } else {
                resp = Response(req->request_id, -1, true);
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
                req->tokens.push_back(resp.token);

                for (auto& kv : req->layer_kv) {
                    kv.set_num_tokens(static_cast<int>(req->tokens.size()));
                }

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
        && finish_queue_.empty()
        && cancel_queue_.empty();
}

void Scheduler::wait_and_enqueue_one_request() {
    
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