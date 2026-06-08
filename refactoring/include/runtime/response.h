#pragma once
#include <iostream>


namespace mini_llm::runtime {
struct Response {
    uint64_t request_id = -1;
    int token;
    bool finished = false;

    Response() = default;

    Response(int req_id, int t, bool done = false)
        : request_id(req_id),
          token(t),
          finished(done) {}
};

}