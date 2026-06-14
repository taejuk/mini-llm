#include "model/gpt2_model.cuh"

namespace mini_llm::model {

std::vector<float> GPT2Model::prefill_logits_for_test(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    prefill(reqs);

    size_t n =
        static_cast<size_t>(reqs.size()) *
        static_cast<size_t>(mini_llm::constants::GPT2_VOCAB_SIZE);

    return std::vector<float>(h_logits, h_logits + n);
}

} // namespace mini_llm::model
