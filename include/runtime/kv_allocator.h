#pragma once

#include "runtime/request.h"

namespace mini_llm::runtime {

class KvAllocator {
public:
    virtual ~KvAllocator() = default;

    virtual bool allocate_prefill(Request& req) = 0;
    virtual bool allocate_decode(Request& req) = 0;
    virtual void free_request(Request& req) = 0;

    virtual bool swap_out(Request& req) = 0;
    virtual bool swap_in(Request& req) = 0;
    virtual bool is_swapped(const Request& req) const = 0;
};

} // namespace mini_llm::runtime