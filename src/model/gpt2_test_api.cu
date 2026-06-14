#include "model/gpt2_model.cuh"

namespace mini_llm::model {

namespace {

std::vector<float> copy_host_logits(
    const float* h_logits,
    size_t batch_size
) {
    size_t n =
        batch_size *
        static_cast<size_t>(mini_llm::constants::GPT2_VOCAB_SIZE);

    return std::vector<float>(h_logits, h_logits + n);
}

} // namespace

std::vector<float> GPT2Model::prefill_logits_for_test(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    prefill(reqs);
    return copy_host_logits(h_logits, reqs.size());
}

std::vector<float> GPT2Model::decode_logits_for_test(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    decode(reqs);
    return copy_host_logits(h_logits, reqs.size());
}

} // namespace mini_llm::model
