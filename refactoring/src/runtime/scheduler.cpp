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
    
}

void Scheduler::run_decode_batch() {
    
}

void Scheduler::run_prefill_batch() {

}

bool Scheduler::no_internal_work() {
    
    return false;
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

        run_decode_batch();

        run_prefill_batch();

        if(no_internal_work()) wait_and_enqueue_one_request();    
        
    }
    // 여기서 호출하면 된다.
}
}