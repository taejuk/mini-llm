#pragma once

#include "runtime/inference_backend.h"
#include "constants.h"

namespace mini_llm::runtime {

class MockBackend final : public InferenceBackend {
public:
    std::vector<Response> prefill(
        std::vector<std::unique_ptr<Request>>& reqs
    ) override {
        std::vector<Response> out;
        out.reserve(reqs.size());

        for (auto& req : reqs) {
            out.emplace_back(req->request_id, 1000, false);
        }

        return out;
    }

    std::vector<Response> decode(
        std::vector<std::unique_ptr<Request>>& reqs
    ) override {
        std::vector<Response> out;
        out.reserve(reqs.size());

        for (auto& req : reqs) {
            int generated = req->generated_tokens();

            if (generated >= req->max_new_tokens - 1) {
                out.emplace_back(
                    req->request_id,
                    mini_llm::constants::GPT2_EOS_TOKEN_ID,
                    true
                );
            } else {
                out.emplace_back(
                    req->request_id,
                    1000 + generated,
                    false
                );
            }
        }

        return out;
    }
};

} // namespace mini_llm::runtime