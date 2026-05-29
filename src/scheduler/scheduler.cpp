#include "scheduler/scheduler.h"
#include <algorithm>
#include <iostream>

Scheduler::Scheduler(int block_size, int hidden_dim)
    : block_size_(block_size), hidden_dim_(hidden_dim)
{}

std::future<std::vector<int>> Scheduler::submit(std::vector<int> prompt_ids,
                                                  int max_new_tokens) {
    auto req = std::make_unique<Request>(
        next_id_++,
        std::move(prompt_ids),
        max_new_tokens,
        block_size_,
        hidden_dim_
    );
    auto fut = req->result_promise.get_future();
    incoming_.push(std::move(req));
    return fut;
}

ScheduleBatch Scheduler::schedule() {
    ScheduleBatch batch;

    std::vector<std::unique_ptr<Request>> tmp;
    incoming_.drain(tmp);
    for (auto& req : tmp) {
        Request* raw = req.get();
        requests_.push_back(std::move(req));  
        waiting_.push_back(raw);              
    }

    try_admit();

    for (Request* req : running_) {
        if (req->state == RequestState::PREFILL) {
            batch.prefill_reqs.push_back(req);
        } else if (req->state == RequestState::DECODE) {
            batch.decode_reqs.push_back(req);
        }
    }

    return batch;
}


void Scheduler::update(const ScheduleBatch& batch,
                       const std::vector<int>& next_tokens) {
    // next_tokens 순서: prefill_reqs 먼저, decode_reqs 다음
    // TODO: 각 요청의 output_ids에 토큰 추가
    // TODO: max_new_tokens 도달 시 state = DONE, result_promise 설정
    // TODO: running에서 DONE 요청 제거
    int token_pos = 0;
    for(auto prefill_req: batch.prefill_reqs) {
        prefill_req->state = RequestState::DECODE;
        prefill_req->output_ids.push_back(next_tokens[token_pos]);
        token_pos++;
    }

    for(auto decode_req: batch.decode_reqs) {
        decode_req->output_ids.push_back(next_tokens[token_pos]);
        token_pos++;
    }
    for(Request* req: running_) {
        if ((int)req->output_ids.size() >= req->max_new_tokens) {
            req->state = RequestState::DONE;
	    
	    for(auto& kv : req->layer_kv) kv.free_all();

            req->result_promise.set_value(req->output_ids);
        }
    }
    running_.erase(
        std::remove_if(running_.begin(), running_.end(),
            [](Request* r){ return r->state == RequestState::DONE;}), running_.end());
}

bool Scheduler::all_done() const {
    
    return waiting_.empty() && running_.empty() && incoming_.empty();
}

int Scheduler::try_admit() {
    int ret = 0;
    int free_blocks = BlockAllocator::getInstance().get_num_free_blocks();
    //std::cout << "[admit] free_blocks=" << free_blocks << " waiting=" << waiting_.size() << "\n";
    free_blocks = free_blocks - blocks_needed_for_running();
    
    while(!waiting_.empty()) {
        Request* cand_rq = waiting_.front();
        int prompt_len = (int)cand_rq->prompt_ids.size();
        int needed = blocks_needed_for(cand_rq);
        if(needed > free_blocks) break;
        cand_rq->state = RequestState::PREFILL;
        running_.push_back(cand_rq);
        waiting_.pop_front();
        free_blocks -= needed;
        ret++;
    }
    return ret;
}

// 다음 running을 실행하기 위해 필요한 block 수
int Scheduler::blocks_needed_for_running() const {
    int needed = 0;

    for (const Request* req : running_) {
        if (req->state != RequestState::DECODE) {
            continue;
        }

        for (const auto& kv : req->layer_kv) {
            const auto& bt = kv.get_block_table();

            if (bt.empty()) {
                needed++;
            } else if (bt.back().filled >= block_size_) {
                needed++;
            }
        }
    }

    return needed;
}

int Scheduler::blocks_needed_for(const Request* req) const {
    int prompt_len = (int)req->prompt_ids.size();
    int per_layer = (prompt_len + block_size_ - 1) / block_size_;
    return per_layer * N_LAYERS;
}
