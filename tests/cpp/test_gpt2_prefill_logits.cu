#include "test_common.cuh"

#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/request.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;
namespace Model = mini_llm::model;

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

namespace {

template <typename T>
std::vector<T> read_all_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);

    if (!ifs) {
        std::cerr << "failed to open " << path << "\n";
        std::cerr << "Generate model golden files first:\n";
        std::cerr << "  python scripts/dump_gpt2_golden.py\n";
        std::exit(1);
    }

    std::streamsize bytes = ifs.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        std::cerr << "invalid binary size: " << path << "\n";
        std::exit(1);
    }

    ifs.seekg(0, std::ios::beg);

    std::vector<T> v(static_cast<size_t>(bytes) / sizeof(T));
    if (!ifs.read(reinterpret_cast<char*>(v.data()), bytes)) {
        std::cerr << "failed to read " << path << "\n";
        std::exit(1);
    }

    return v;
}

std::string model_case_path(const std::string& file) {
    return std::string(MODEL_TEST_DATA_DIR) + "/" + file;
}

int argmax(const std::vector<float>& x) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(x.size()); ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return best;
}

bool contains_topk(const std::vector<float>& x, int target, int k) {
    std::vector<int> idx(x.size());
    for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
        idx[i] = i;
    }

    std::partial_sort(
        idx.begin(),
        idx.begin() + std::min(k, static_cast<int>(idx.size())),
        idx.end(),
        [&](int a, int b) { return x[a] > x[b]; }
    );

    int limit = std::min(k, static_cast<int>(idx.size()));
    for (int i = 0; i < limit; ++i) {
        if (idx[i] == target) {
            return true;
        }
    }

    return false;
}

struct LogitMetrics {
    double mean_abs = 0.0;
    float max_abs = 0.0f;
    double cosine = 0.0;
};

LogitMetrics compare_logits(
    const std::vector<float>& got,
    const std::vector<float>& expected
) {
    if (got.size() != expected.size()) {
        std::cerr << "logit size mismatch: got=" << got.size()
                  << " expected=" << expected.size() << "\n";
        std::exit(1);
    }

    double sum_abs = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double expected_norm = 0.0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::abs(got[i] - expected[i]);
        sum_abs += diff;
        max_abs = std::max(max_abs, diff);

        dot += static_cast<double>(got[i]) * expected[i];
        got_norm += static_cast<double>(got[i]) * got[i];
        expected_norm += static_cast<double>(expected[i]) * expected[i];
    }

    LogitMetrics m;
    m.mean_abs = sum_abs / static_cast<double>(got.size());
    m.max_abs = max_abs;
    m.cosine = dot / (std::sqrt(got_norm) * std::sqrt(expected_norm) + 1e-12);
    return m;
}

void prepare_prefill_kv_blocks(Rt::Request& req) {
    int blocks_per_layer =
        (req.prompts_len + C::DEFAULT_KV_BLOCK_SIZE - 1) /
        C::DEFAULT_KV_BLOCK_SIZE;

    for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
        Rt::PagedKVCache& kv = req.layer_kv[layer];
        kv.reset();

        for (int b = 0; b < blocks_per_layer; ++b) {
            // This test only validates prefill logits. The prefill path writes
            // KV to the pool but computes attention from buf_qkv directly, so
            // deterministic non-overlapping block ids are sufficient here.
            kv.append_block(layer * blocks_per_layer + b);
        }
    }
}

} // namespace

int main() {
    auto input_ids = read_all_bin<int32_t>(model_case_path("input_ids.bin"));
    auto expected_logits = read_all_bin<float>(model_case_path("expected_logits.bin"));

    if (input_ids.empty()) {
        std::cerr << "input_ids.bin is empty\n";
        return 1;
    }

    if (expected_logits.size() != static_cast<size_t>(C::GPT2_VOCAB_SIZE)) {
        std::cerr << "expected_logits.bin must contain exactly GPT2_VOCAB_SIZE floats, got "
                  << expected_logits.size() << "\n";
        return 1;
    }

    std::vector<int> tokens(input_ids.begin(), input_ids.end());

    std::vector<std::unique_ptr<Rt::Request>> reqs;
    reqs.emplace_back(std::make_unique<Rt::Request>(0, tokens, 1));
    prepare_prefill_kv_blocks(*reqs[0]);

    auto& model = Model::GPT2Model::get();
    std::vector<float> got_logits = model.prefill_logits_for_test(reqs);

    if (got_logits.size() != static_cast<size_t>(C::GPT2_VOCAB_SIZE)) {
        std::cerr << "got logits must contain exactly GPT2_VOCAB_SIZE floats, got "
                  << got_logits.size() << "\n";
        return 1;
    }

    LogitMetrics m = compare_logits(got_logits, expected_logits);

    int got_top1 = argmax(got_logits);
    int expected_top1 = argmax(expected_logits);
    bool top1_match = got_top1 == expected_top1;
    bool top5_hit = contains_topk(got_logits, expected_top1, 5);

    std::cout << "gpt2_prefill_logits mean_abs_error = " << m.mean_abs << "\n";
    std::cout << "gpt2_prefill_logits max_abs_error  = " << m.max_abs << "\n";
    std::cout << "gpt2_prefill_logits cosine_sim     = " << m.cosine << "\n";
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
        std::cerr << "[FAIL] gpt2_prefill_logits\n";
        return 1;
    }

    std::cout << "[PASS] gpt2_prefill_logits\n";
    return 0;
}
