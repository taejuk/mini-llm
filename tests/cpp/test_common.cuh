#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(1);                                                     \
        }                                                                    \
    } while (0)

inline std::string case_path(const std::string& kernel, const std::string& file) {
    return std::string(TEST_DATA_DIR) + "/" + kernel + "/" + file;
}

template <typename T>
std::vector<T> read_bin(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream ifs(path, std::ios::binary);

    if (!ifs) {
        std::cerr << "failed to open " << path << "\n";
        std::exit(1);
    }

    ifs.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));

    if (!ifs) {
        std::cerr << "failed to read " << path << "\n";
        std::exit(1);
    }

    return v;
}

inline float max_abs_error(
    const std::vector<float>& got,
    const std::vector<float>& expected
) {
    if (got.size() != expected.size()) {
        std::cerr << "size mismatch: got=" << got.size()
                  << " expected=" << expected.size() << "\n";
        std::exit(1);
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        max_err = std::max(max_err, std::abs(got[i] - expected[i]));
    }
    return max_err;
}

inline void check_close(
    const std::string& name,
    const std::vector<float>& got,
    const std::vector<float>& expected,
    float tol
) {
    float err = max_abs_error(got, expected);

    std::cout << name << " max_abs_error = " << err << "\n";

    if (err > tol || std::isnan(err)) {
        std::cerr << "[FAIL] " << name << "\n";
        std::exit(1);
    }

    std::cout << "[PASS] " << name << "\n";
}