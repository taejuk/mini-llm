#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace mini_llm::profiling {

inline bool profile_enabled(const char* env_name) {
    return std::getenv(env_name) != nullptr;
}

inline void print_profile_header_once() {
    static bool printed = false;

    if (!printed) {
        std::cout
            << "kind,batch_size,total_tokens,layer,stage,ms\n";
        printed = true;
    }
}

inline void sync_or_die(const char* where) {
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr
            << "[profile] cudaDeviceSynchronize failed at "
            << where << ": "
            << cudaGetErrorString(err)
            << "\n";
        std::exit(1);
    }
}

class ScopedStageTimer {
public:
    ScopedStageTimer(
        const char* env_name,
        const char* kind,
        int batch_size,
        std::size_t total_tokens,
        int layer,
        const char* stage
    )
        : enabled_(profile_enabled(env_name)),
          kind_(kind),
          batch_size_(batch_size),
          total_tokens_(total_tokens),
          layer_(layer),
          stage_(stage) {
        if (!enabled_) {
            return;
        }

        print_profile_header_once();
        sync_or_die("timer_start");
        start_ = Clock::now();
    }

    ~ScopedStageTimer() {
        if (!enabled_) {
            return;
        }

        sync_or_die("timer_stop");
        auto end = Clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(
                end - start_
            ).count();

        std::cout
            << kind_ << ","
            << batch_size_ << ","
            << total_tokens_ << ","
            << layer_ << ","
            << stage_ << ","
            << std::fixed << std::setprecision(4)
            << ms << "\n";
    }

    ScopedStageTimer(const ScopedStageTimer&) = delete;
    ScopedStageTimer& operator=(const ScopedStageTimer&) = delete;

private:
    using Clock = std::chrono::high_resolution_clock;

    bool enabled_;
    const char* kind_;
    int batch_size_;
    std::size_t total_tokens_;
    int layer_;
    const char* stage_;
    Clock::time_point start_;
};

} // namespace mini_llm::profiling
