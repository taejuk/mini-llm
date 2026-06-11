#include "runtime/scheduler.h"


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
// 여기에 로직이 담겨져 있어야 한다.
void Scheduler::worker_loop() {
    // 여기서 model을 가지고 와야 한다.
    while (running_.load()) {

        if(decode_queue_.empty() && prefill_queue_.empty()) {
            std::optional<std::unique_ptr<Request>> req_opt = request_queue_.wait_pop();
            if(!req_opt.has_value()) break;
            std::vector<std::unique_ptr<Request>> reqs;
            reqs.push_back(std::move(req_opt.value()));
            request_queue_.drain(reqs);
            std::cout << "requests: " << reqs.size() << std::endl;
            for(int i = 0; i < reqs.size(); i++) prefill_queue_.push(std::move(reqs[i]));

        }
        int i = 0;
        int max_i = decode_queue_.size();
        while(i < max_i) {
            // 하나씩 꺼내서 decode를 실행할 수 있는지 평가해야 한다.

            // 
        }
        
        // prefill에 대해서 처리한다.
        //while(!prefill_queue_.empty()) {}
        for(int i = 0; i < prefill_queue_.size(); i++) {
            // BATCH NUM까지 꺼낸다.

            // batch_prefill을 실행한다.

            // response를 보고 done인지 아닌지 판단한 후에,
            // done이면 free하고 new_decode_queue 안 넣는다.

            // done이 아니면 다시 new_decode_queue 넣는다.
        }
        // 들어온 response들을 drain한다.
        
        std::optional<std::unique_ptr<Request>> req_opt = request_queue_.wait_pop();
        if(!req_opt.has_value()) break;
        std::vector<std::unique_ptr<Request>> reqs;
        reqs.push_back(std::move(req_opt.value()));
        request_queue_.drain(reqs);
        for(int i = 0; i < reqs.size(); i++) prefill_queue_.push(std::move(reqs[i]));
    }
    // 여기서 호출하면 된다.
}
}