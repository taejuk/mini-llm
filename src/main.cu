#include "server/http_server.h"
#include "scheduler/scheduler.h"
#include "model/gpt2_wmma.cuh"
#include "util/parser.h"

#include <thread>
#include <chrono>
#include <iostream>

static constexpr int BLOCK_SIZE = 16;

// ── 추론 루프 (별도 스레드) ───────────────────────────────────────
// epoll thread와 독립적으로 돌면서 schedule → GPU forward → update 반복
void inference_loop(Scheduler& sched) {
    auto& model = GPT2ModelWMMA::get();

    while (true) {
        ScheduleBatch batch = sched.schedule();

        // 처리할 요청 없으면 잠깐 대기
        if (batch.prefill_reqs.empty() && batch.decode_reqs.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        std::vector<int> next_tokens;

        // prefill: 요청마다 순서대로 처리
        for (Request* req : batch.prefill_reqs) {
            int* d_ids;
            cudaMalloc(&d_ids, req->prompt_ids.size() * sizeof(int));
            cudaMemcpy(d_ids, req->prompt_ids.data(),
                       req->prompt_ids.size() * sizeof(int),
                       cudaMemcpyHostToDevice);

            int next_tok = model.prefill(d_ids, (int)req->prompt_ids.size(), req->kv);
            next_tokens.push_back(next_tok);
            cudaFree(d_ids);
        }

        // batch decode: decode_reqs 전체를 한 번에 처리
        if (!batch.decode_reqs.empty()) {
            auto decode_toks = model.batch_decode(batch.decode_reqs);
            for (int tok : decode_toks) next_tokens.push_back(tok);
        }

        // 결과 반영 → DONE된 요청은 promise 설정
        sched.update(batch, next_tokens);
    }
}

int main() {
    
    static constexpr int TOTAL_BLOCKS = 128;
    BlockAllocator::getInstance(TOTAL_BLOCKS, BLOCK_SIZE, D_MODEL);
    std::cout << "[main] allocator ready: " << TOTAL_BLOCKS << " blocks\n";

    // 1. GPT-2 모델 초기화 (weights 로딩)
    std::cout << "[main] loading weights from " << WEIGHTS_DIR << "\n";
    GPT2ModelWMMA::init(WEIGHTS_DIR, BLOCK_SIZE);
    std::cout << "[main] model ready\n";

    // 2. Scheduler 생성
    Scheduler sched(BLOCK_SIZE, D_MODEL);

    // 3. 추론 루프 스레드 시작 (epoll thread와 별개)
    std::thread infer_thread(inference_loop, std::ref(sched));
    infer_thread.detach();
    std::cout << "[main] inference thread started\n";

    // 4. HTTP 서버 시작 (blocking)
    HttpServer server(8080, [&sched](const HttpRequest& req) -> HttpResponse {
        try {
            InferRequest infer = parse_request(req.body);
            auto fut = sched.submit(infer.input_ids, infer.max_tokens);
            auto output_ids = fut.get();  // watcher thread에서 호출 → 블로킹 OK
            return { 200, make_response(output_ids) };
        } catch (const std::exception& e) {
            return { 400, std::string("{\"error\": \"") + e.what() + "\"}" };
        }
    });

    server.run();
    return 0;
}
