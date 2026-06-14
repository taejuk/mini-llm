#pragma once

#include "runtime/inference_backend.h"
#include "model/gpt2_model.cuh"

namespace mini_llm::runtime {

class Gpt2Backend final : public InferenceBackend {
private:
    mini_llm::model::GPT2Model& model_;

public:
    Gpt2Backend()
        : model_(mini_llm::model::GPT2Model::get()) {}

    std::vector<Response> prefill(
        std::vector<std::unique_ptr<Request>>& reqs
    ) override {
        return model_.prefill(reqs);
    }

    std::vector<Response> decode(
        std::vector<std::unique_ptr<Request>>& reqs
    ) override {
        return model_.decode(reqs);
    }
};

} // namespace mini_llm::runtime