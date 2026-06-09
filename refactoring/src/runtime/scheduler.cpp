#include "runtime/scheduler.h"


namespace mini_llm::runtime {
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

        // decode에 대해서 먼저 처리한다.
        // 이 때 empty가 아니라 for문으로 한번 돈다.
        //while(!decode_queue_.empty()){}
        
        // prefill에 대해서 처리한다.
        //while(!prefill_queue_.empty()) {}
        for(int i = 0; i < )
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