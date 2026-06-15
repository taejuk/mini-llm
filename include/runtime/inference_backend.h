#pragma once

#include <memory>
#include <vector>

#include "runtime/request.h"
#include "runtime/response.h"

namespace mini_llm::runtime {

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual std::vector<Response> prefill(
        std::vector<std::unique_ptr<Request>>& reqs
    ) = 0;

    virtual std::vector<Response> decode(
        std::vector<std::unique_ptr<Request>>& reqs
    ) = 0;

    
};

} // namespace mini_llm::runtime