#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>

enum class TensorDType {
    Float32,
    Int32,
    Int64,
};

struct TensorDumpConfig {
    bool enabled = false;
    std::string output_dir = "mini_ref";

    // 비어 있으면 모든 tensor 저장
    std::unordered_set<std::string> allowlist;
};

class TensorDumper {
public:
    explicit TensorDumper(TensorDumpConfig config);

    bool enabled() const;
    bool should_dump(const std::string& name) const;

    void dump_device_float(
        const std::string& name,
        const float* d_ptr,
        const std::vector<int64_t>& shape,
        cudaStream_t stream = nullptr
    );

    void dump_device_int32(
        const std::string& name,
        const int32_t* d_ptr,
        const std::vector<int64_t>& shape,
        cudaStream_t stream = nullptr
    );

    void dump_host_float(
        const std::string& name,
        const float* h_ptr,
        const std::vector<int64_t>& shape
    );

    void dump_host_int32(
        const std::string& name,
        const int32_t* h_ptr,
        const std::vector<int64_t>& shape
    );

private:
    TensorDumpConfig config_;

private:
    static size_t numel(const std::vector<int64_t>& shape);
    static const char* dtype_name(TensorDType dtype);
    static size_t dtype_size(TensorDType dtype);

    void ensure_output_dir() const;

    void write_bin(
        const std::string& name,
        const void* data,
        size_t bytes
    ) const;

    void write_shape(
        const std::string& name,
        TensorDType dtype,
        const std::vector<int64_t>& shape
    ) const;

    void dump_host_raw(
        const std::string& name,
        const void* h_ptr,
        TensorDType dtype,
        const std::vector<int64_t>& shape
    );

    void dump_device_raw(
        const std::string& name,
        const void* d_ptr,
        TensorDType dtype,
        const std::vector<int64_t>& shape,
        cudaStream_t stream
    );
};