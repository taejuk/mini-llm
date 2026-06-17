#include "constants.h"
#include "model/gpt2_model.cuh"
#include "runtime/pagekvcache.h"
#include "runtime/request.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace C = mini_llm::constants;
namespace Rt = mini_llm::runtime;
namespace Model = mini_llm::model;

namespace {

template <typename T>
std::vector<T> read_all_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);

    if (!ifs) {
        std::cerr << "failed to open " << path << "\n";
        std::cerr << "Generate batch decode golden files first:\n";
        std::cerr << "  python scripts/dump_gpt2_batch_decode_golden.py\n";
        std::exit(1);
    }

    std::streamsize bytes = ifs.tellg();

    if (bytes < 0 ||
        bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
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

struct LogitMetrics {
    double mean_abs = 0.0;
    float max_abs = 0.0f;
    double cosine = 0.0;
};

LogitMetrics compare_row_logits(
    const std::vector<float>& got,
    const std::vector<float>& expected,
    int row,
    int vocab_size
) {
    size_t offset = static_cast<size_t>(row) * vocab_size;

    double sum_abs = 0.0;
    double dot = 0.0;
    double got_norm = 0.0;
    double expected_norm = 0.0;
    float max_abs = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        float g = got[offset + i];
        float e = expected[offset + i];
        float diff = std::abs(g - e);

        sum_abs += diff;
        max_abs = std::max(max_abs, diff);

        dot += static_cast<double>(g) * e;
        got_norm += static_cast<double>(g) * g;
        expected_norm += static_cast<double>(e) * e;
    }

    LogitMetrics m;
    m.mean_abs = sum_abs / static_cast<double>(vocab_size);
    m.max_abs = max_abs;
    m.cosine =
        dot /
        (std::sqrt(got_norm) * std::sqrt(expected_norm) + 1e-12);

    return m;
}

int argmax_row(
    const std::vector<float>& x,
    int row,
    int vocab_size
) {
    size_t offset = static_cast<size_t>(row) * vocab_size;

    int best = 0;

    for (int i = 1; i < vocab_size; ++i) {
        if (x[offset + i] > x[offset + best]) {
            best = i;
        }
    }

    return best;
}

bool contains_topk_row(
    const std::vector<float>& x,
    int row,
    int vocab_size,
    int target,
    int k
) {
    size_t offset = static_cast<size_t>(row) * vocab_size;

    std::vector<int> idx(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        idx[i] = i;
    }

    int limit = std::min(k, vocab_size);

    std::partial_sort(
        idx.begin(),
        idx.begin() + limit,
        idx.end(),
        [&](int a, int b) {
            return x[offset + a] > x[offset + b];
        }
    );

    for (int i = 0; i < limit; ++i) {
        if (idx[i] == target) {
            return true;
        }
    }

    return false;
}

std::vector<std::unique_ptr<Rt::Request>> make_requests_from_flat_tokens(
    const std::vector<int32_t>& flat_input_ids,
    const std::vector<int32_t>& prompt_lens
) {
    std::vector<std::unique_ptr<Rt::Request>> reqs;
    reqs.reserve(prompt_lens.size());

    size_t offset = 0;

    for (size_t i = 0; i < prompt_lens.size(); ++i) {
        int len = prompt_lens[i];

        if (len <= 0) {
            std::cerr << "invalid prompt length at row "
                      << i << ": " << len << "\n";
            std::exit(1);
        }

        if (len + 1 >= C::MAX_SEQ) {
            std::cerr
                << "prompt length is not safe for one decode step at row "
                << i << ": " << len << "\n";
            std::exit(1);
        }

        if (offset + static_cast<size_t>(len) > flat_input_ids.size()) {
            std::cerr << "input_ids length mismatch while reading row "
                      << i << "\n";
            std::exit(1);
        }

        std::vector<int> tokens;
        tokens.reserve(len + 1);

        for (int j = 0; j < len; ++j) {
            tokens.push_back(
                static_cast<int>(flat_input_ids[offset + j])
            );
        }

        offset += static_cast<size_t>(len);

        reqs.emplace_back(
            std::make_unique<Rt::Request>(
                static_cast<int>(i),
                std::move(tokens),
                1
            )
        );
    }

    if (offset != flat_input_ids.size()) {
        std::cerr << "unused input tokens: used=" << offset
                  << " total=" << flat_input_ids.size() << "\n";
        std::exit(1);
    }

    return reqs;
}

void prepare_batched_decode_kv_blocks(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    int next_block_id = 0;

    for (auto& req : reqs) {
        // Need one extra token position for decode.
        int total_tokens_needed = req->prompts_len + 1;

        int blocks_per_layer =
            (total_tokens_needed + C::DEFAULT_KV_BLOCK_SIZE - 1) /
            C::DEFAULT_KV_BLOCK_SIZE;

        for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
            Rt::PagedKVCache& kv = req->layer_kv[layer];
            kv.reset();

            for (int b = 0; b < blocks_per_layer; ++b) {
                if (next_block_id >= C::DEFAULT_TOTAL_KV_BLOCKS) {
                    std::cerr
                        << "test needs too many KV blocks: "
                        << next_block_id + 1
                        << " > "
                        << C::DEFAULT_TOTAL_KV_BLOCKS
                        << "\n";
                    std::exit(1);
                }

                kv.append_block(next_block_id++);
            }
        }
    }
}

void mark_prefill_cached(
    std::vector<std::unique_ptr<Rt::Request>>& reqs
) {
    for (auto& req : reqs) {
        for (int layer = 0; layer < C::GPT2_N_LAYERS; ++layer) {
            req->layer_kv[layer].set_num_tokens(req->prompts_len);
        }
    }
}

void append_decode_tokens(
    std::vector<std::unique_ptr<Rt::Request>>& reqs,
    const std::vector<int32_t>& decode_tokens
) {
    if (decode_tokens.size() != reqs.size()) {
        std::cerr << "decode token count mismatch: got="
                  << decode_tokens.size()
                  << " expected="
                  << reqs.size()
                  << "\n";
        std::exit(1);
    }

    for (size_t i = 0; i < reqs.size(); ++i) {
        reqs[i]->tokens.push_back(
            static_cast<int>(decode_tokens[i])
        );

        reqs[i]->state = Rt::RequestState::DecodeReady;
        reqs[i]->kv_residency = Rt::KvCacheResidency::Gpu;
    }
}

} // namespace

int main() {
    auto flat_input_ids =
        read_all_bin<int32_t>(
            model_case_path("batch_decode_input_ids.bin")
        );

    auto prompt_lens =
        read_all_bin<int32_t>(
            model_case_path("batch_decode_prompt_lens.bin")
        );

    auto decode_tokens =
        read_all_bin<int32_t>(
            model_case_path("batch_decode_tokens.bin")
        );

    auto expected_logits =
        read_all_bin<float>(
            model_case_path("batch_decode_expected_logits.bin")
        );

    if (prompt_lens.empty()) {
        std::cerr << "batch_decode_prompt_lens.bin is empty\n";
        return 1;
    }

    int batch_size = static_cast<int>(prompt_lens.size());

    if (batch_size > C::MAX_BATCH_NUM) {
        std::cerr << "batch_size=" << batch_size
                  << " exceeds MAX_BATCH_NUM="
                  << C::MAX_BATCH_NUM << "\n";
        return 1;
    }

    if (decode_tokens.size() != static_cast<size_t>(batch_size)) {
        std::cerr << "batch_decode_tokens.bin size mismatch: got="
                  << decode_tokens.size()
                  << " expected="
                  << batch_size
                  << "\n";
        return 1;
    }

    size_t expected_logit_count =
        static_cast<size_t>(batch_size) *
        C::GPT2_VOCAB_SIZE;

    if (expected_logits.size() != expected_logit_count) {
        std::cerr
            << "batch_decode_expected_logits.bin size mismatch: got="
            << expected_logits.size()
            << " expected="
            << expected_logit_count
            << "\n";
        return 1;
    }

    auto reqs = make_requests_from_flat_tokens(
        flat_input_ids,
        prompt_lens
    );

    // Allocate enough KV block ids for both prefill KV and the one decode token.
    // This is equivalent to prefill allocation + decode allocation.
    prepare_batched_decode_kv_blocks(reqs);

    auto& model = Model::GPT2Model::get();

    // 1. Batched prefill fills KV pool.
    (void)model.prefill_logits_for_test(reqs);

    // 2. decode() expects KV metadata to say prompt tokens are already cached.
    mark_prefill_cached(reqs);

    // 3. Append request-specific HF decode token.
    append_decode_tokens(reqs, decode_tokens);

    // 4. Batched decode logits.
    std::vector<float> got_logits =
        model.decode_logits_for_test(reqs);

    if (got_logits.size() != expected_logit_count) {
        std::cerr << "got logits size mismatch: got="
                  << got_logits.size()
                  << " expected="
                  << expected_logit_count
                  << "\n";
        return 1;
    }

    constexpr double MEAN_ABS_TOL = 1e-2;
    constexpr float MAX_ABS_TOL = 2e-1f;
    constexpr double COSINE_TOL = 0.999;

    bool all_pass = true;

    double total_mean_abs = 0.0;
    float global_max_abs = 0.0f;
    double min_cosine = 1.0;

    std::cout << "batch_size = " << batch_size << "\n";

    for (int row = 0; row < batch_size; ++row) {
        LogitMetrics m = compare_row_logits(
            got_logits,
            expected_logits,
            row,
            C::GPT2_VOCAB_SIZE
        );

        int got_top1 = argmax_row(
            got_logits,
            row,
            C::GPT2_VOCAB_SIZE
        );

        int expected_top1 = argmax_row(
            expected_logits,
            row,
            C::GPT2_VOCAB_SIZE
        );

        bool top1_match = got_top1 == expected_top1;

        bool top5_hit = contains_topk_row(
            got_logits,
            row,
            C::GPT2_VOCAB_SIZE,
            expected_top1,
            5
        );

        bool row_pass =
            top1_match &&
            top5_hit &&
            m.mean_abs <= MEAN_ABS_TOL &&
            m.max_abs <= MAX_ABS_TOL &&
            m.cosine >= COSINE_TOL &&
            !std::isnan(m.mean_abs) &&
            !std::isnan(m.max_abs) &&
            !std::isnan(m.cosine);

        total_mean_abs += m.mean_abs;
        global_max_abs = std::max(global_max_abs, m.max_abs);
        min_cosine = std::min(min_cosine, m.cosine);

        std::cout
            << "row=" << row
            << " prompt_len=" << prompt_lens[row]
            << " decode_token=" << decode_tokens[row]
            << " mean_abs_error=" << m.mean_abs
            << " max_abs_error=" << m.max_abs
            << " cosine_sim=" << m.cosine
            << " expected_top1=" << expected_top1
            << " got_top1=" << got_top1
            << " top1_match=" << (top1_match ? "true" : "false")
            << " top5_hit=" << (top5_hit ? "true" : "false")
            << "\n";

        if (!row_pass) {
            all_pass = false;
        }
    }

    double avg_mean_abs =
        total_mean_abs / static_cast<double>(batch_size);

    std::cout << "batch_decode avg_mean_abs_error = "
              << avg_mean_abs << "\n";
    std::cout << "batch_decode global_max_abs     = "
              << global_max_abs << "\n";
    std::cout << "batch_decode min_cosine         = "
              << min_cosine << "\n";

    if (!all_pass) {
        std::cerr << "[FAIL] gpt2_batch_decode_logits\n";
        return 1;
    }

    std::cout << "[PASS] gpt2_batch_decode_logits\n";
    return 0;
}