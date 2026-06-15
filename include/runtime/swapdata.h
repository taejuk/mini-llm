#pragma once

#include <vector>

namespace mini_llm::runtime {
struct SwappedBlock {
    int cpu_block_id = -1;
    int valid_tokens = 0;
};

struct SwappedRequest {
    std::vector<SwappedBlock> cpu_blocks; // layer-major order
    int blocks_per_layer = 0;
};

}