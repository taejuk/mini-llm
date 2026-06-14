#include "test_common.cuh"

#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/request.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;
namespace Model = mini_llm::model;

namespace {

constexpr int PROMPT_LEN = 5;

int argmax(const std::vector<float>& x) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(x.size()); ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

bool topk_has(const std::vector<float>& x, int target, int k) {
    std::vector<int> idx(x.size());
    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
        idx[i] = i;
    }

    std::partial_sort(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [&](int a, int b) { return x[a] > x[b]; }
    );

    for (int i = 0; i < k; ++i) {
        if (idx[i] == target) {
            return true;
        }
    }

    return false;
}

struct Metrics {
    double mean_abs = 0.0;
    float max_abs = 0.0f;
    double cosine = 0.0;
};

Metrics compare_logits(
    const std::vector<float>& got,
    const std::vector<float>& ref
) {
    if (got.size() != ref.size()) {
        std::cerr << "size mismatch\n";
        std::exit(1);
    }

    double sum_abs = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double ref_norm = 0.0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::abs(got[i] - ref[i]);

        sum_abs += diff;
        max_abs = std::max(max_abs, diff);

        dot += static_cast<double>(got[i]) * ref[i];
        got_norm += static_cast<double>(got[i]) * got[i];
        ref_norm += static_cast<double>(ref[i]) * ref[i];
    }

    Metrics m;
    m.mean_abs = sum_abs / static_cast<double>(got.size());
    m.max_abs = max_abs;
    m.cosine = dot / (std::sqrt(got_norm) * std::sqrt(ref_norm) + 1e-12);
    return m;
}

void prepare_kv_blocks(Rt::Request& req, int total_tokens_needed) {
    int blocks_per_layer =
        (total_tokens_needed + C::DEFAULT_KV_BLOCK_SIZE - 1) /
        C::DEFAULT_KV_BLOCK_SIZE;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        auto& kv = req.layer_kv[layer];
        kv.reset();

        for (int b = 0; b < blocks_per_layer; ++b) {
            kv.append_block(layer * blocks_per_layer + b);
        }
    }
}

void mark_prefill_cached(Rt::Request& req) {
    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        req.layer_kv[layer].set_num_tokens(req.prompts_len);
    }
}

} // namespace

int main() {
    auto input_ids = read_bin<int32_t>(
        std::string(MODEL_TEST_DATA_DIR) + "/input_ids.bin",
        PROMPT_LEN
    );

    auto decode_token = read_bin<int32_t>(
        std::string(MODEL_TEST_DATA_DIR) + "/decode_token.bin",
        1
    );

    auto expected = read_bin<float>(
        std::string(MODEL_TEST_DATA_DIR) + "/expected_decode_logits.bin",
        C::GPT2_VOCAB_SIZE
    );

    std::vector<int> prompt(input_ids.begin(), input_ids.end());

    std::vector<std::unique_ptr<Rt::Request>> reqs;
    reqs.emplace_back(std::make_unique<Rt::Request>(0, prompt, 1));

    prepare_kv_blocks(*reqs[0], PROMPT_LEN + 1);

    auto& model = Model::GPT2Model::get();

    // 1. prefill로 KV pool을 채운다.
    (void)model.prefill_logits_for_test(reqs);

    // 2. decode()가 기대하는 metadata를 맞춘다.
    mark_prefill_cached(*reqs[0]);

    // 3. HF golden에서 사용한 1-step input token을 넣는다.
    reqs[0]->tokens.push_back(static_cast<int>(decode_token[0]));

    // 4. decode 1-step logits 비교.
    std::vector<float> got = model.decode_logits_for_test(reqs);

    Metrics m = compare_logits(got, expected);

    int got_top1 = argmax(got);
    int expected_top1 = argmax(expected);
    bool top1_match = got_top1 == expected_top1;
    bool top5_hit = topk_has(got, expected_top1, 5);

    std::cout << "gpt2_decode_logits mean_abs_error = " << m.mean_abs << "\n";
    std::cout << "gpt2_decode_logits max_abs_error  = " << m.max_abs << "\n";
    std::cout << "gpt2_decode_logits cosine_sim     = " << m.cosine << "\n";
    std::cout << "decode_token  = " << decode_token[0] << "\n";
    std::cout << "expected_top1 = " << expected_top1 << "\n";
    std::cout << "got_top1      = " << got_top1 << "\n";
    std::cout << "top1_match    = " << (top1_match ? "true" : "false") << "\n";
    std::cout << "top5_hit       = " << (top5_hit ? "true" : "false") << "\n";

    constexpr double MEAN_ABS_TOL = 1e-2;
    constexpr float MAX_ABS_TOL = 2e-1f;
    constexpr double COSINE_TOL = 0.999;

    if (!top1_match ||
        !top5_hit ||
        m.mean_abs > MEAN_ABS_TOL ||
        m.max_abs > MAX_ABS_TOL ||
        m.cosine < COSINE_TOL ||
        std::isnan(m.mean_abs) ||
        std::isnan(m.max_abs) ||
        std::isnan(m.cosine)) {
        std::cerr << "[FAIL] gpt2_decode_logits\n";
        return 1;
    }

    std::cout << "[PASS] gpt2_decode_logits\n";
    return 0;
}