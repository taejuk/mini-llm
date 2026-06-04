#include "tensor_dumper.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "[TensorDumper] CUDA error in " << what
                  << ": " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace

TensorDumper::TensorDumper(TensorDumpConfig config)
    : config_(std::move(config)) {
    if (config_.enabled) {
        ensure_output_dir();
    }
}

bool TensorDumper::enabled() const {
    return config_.enabled;
}

bool TensorDumper::should_dump(const std::string& name) const {
    if (!config_.enabled) {
        return false;
    }

    if (config_.allowlist.empty()) {
        return true;
    }

    return config_.allowlist.find(name) != config_.allowlist.end();
}

size_t TensorDumper::numel(const std::vector<int64_t>& shape) {
    size_t n = 1;

    for (int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("[TensorDumper] invalid shape dimension");
        }

        n *= static_cast<size_t>(dim);
    }

    return n;
}

const char* TensorDumper::dtype_name(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::Float32:
            return "float32";
        case TensorDType::Int32:
            return "int32";
        case TensorDType::Int64:
            return "int64";
        default:
            return "unknown";
    }
}

size_t TensorDumper::dtype_size(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::Float32:
            return sizeof(float);
        case TensorDType::Int32:
            return sizeof(int32_t);
        case TensorDType::Int64:
            return sizeof(int64_t);
        default:
            throw std::runtime_error("[TensorDumper] unknown dtype");
    }
}

void TensorDumper::ensure_output_dir() const {
    std::filesystem::create_directories(config_.output_dir);
}

void TensorDumper::write_bin(
    const std::string& name,
    const void* data,
    size_t bytes
) const {
    const std::string path = config_.output_dir + "/" + name + ".bin";

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("[TensorDumper] failed to open " + path);
    }

    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));

    if (!ofs) {
        throw std::runtime_error("[TensorDumper] failed to write " + path);
    }
}

void TensorDumper::write_meta(
    const std::string& name,
    TensorDType dtype,
    const std::vector<int64_t>& shape
) const {
    const std::string path = config_.output_dir + "/" + name + ".shape";

    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("[TensorDumper] failed to open " + path);
    }

    ofs << dtype_name(dtype) << "\n";

    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) {
            ofs << " ";
        }
        ofs << shape[i];
    }

    ofs << "\n";
}

void TensorDumper::dump_host_raw(
    const std::string& name,
    const void* h_ptr,
    TensorDType dtype,
    const std::vector<int64_t>& shape
) {
    if (!should_dump(name)) {
        return;
    }

    if (h_ptr == nullptr) {
        throw std::runtime_error("[TensorDumper] null host pointer: " + name);
    }

    ensure_output_dir();

    const size_t n = numel(shape);
    const size_t bytes = n * dtype_size(dtype);

    write_bin(name, h_ptr, bytes);
    write_meta(name, dtype, shape);

    std::cerr << "[TensorDumper] dumped host tensor "
              << name << " numel=" << n
              << " bytes=" << bytes << "\n";
}

void TensorDumper::dump_device_raw(
    const std::string& name,
    const void* d_ptr,
    TensorDType dtype,
    const std::vector<int64_t>& shape,
    cudaStream_t stream
) {
    if (!should_dump(name)) {
        return;
    }

    if (d_ptr == nullptr) {
        throw std::runtime_error("[TensorDumper] null device pointer: " + name);
    }

    ensure_output_dir();

    const size_t n = numel(shape);
    const size_t bytes = n * dtype_size(dtype);

    std::vector<uint8_t> host(bytes);

    if (stream != nullptr) {
        check_cuda(
            cudaMemcpyAsync(
                host.data(),
                d_ptr,
                bytes,
                cudaMemcpyDeviceToHost,
                stream
            ),
            "cudaMemcpyAsync DeviceToHost"
        );

        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    } else {
        check_cuda(
            cudaMemcpy(
                host.data(),
                d_ptr,
                bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy DeviceToHost"
        );
    }

    write_bin(name, host.data(), bytes);
    write_meta(name, dtype, shape);

    std::cerr << "[TensorDumper] dumped device tensor "
              << name << " numel=" << n
              << " bytes=" << bytes << "\n";
}

void TensorDumper::dump_device_float(
    const std::string& name,
    const float* d_ptr,
    const std::vector<int64_t>& shape,
    cudaStream_t stream
) {
    dump_device_raw(name, d_ptr, TensorDType::Float32, shape, stream);
}

void TensorDumper::dump_device_int32(
    const std::string& name,
    const int32_t* d_ptr,
    const std::vector<int64_t>& shape,
    cudaStream_t stream
) {
    dump_device_raw(name, d_ptr, TensorDType::Int32, shape, stream);
}

void TensorDumper::dump_device_int64(
    const std::string& name,
    const int64_t* d_ptr,
    const std::vector<int64_t>& shape,
    cudaStream_t stream
) {
    dump_device_raw(name, d_ptr, TensorDType::Int64, shape, stream);
}

void TensorDumper::dump_host_float(
    const std::string& name,
    const float* h_ptr,
    const std::vector<int64_t>& shape
) {
    dump_host_raw(name, h_ptr, TensorDType::Float32, shape);
}

void TensorDumper::dump_host_int32(
    const std::string& name,
    const int32_t* h_ptr,
    const std::vector<int64_t>& shape
) {
    dump_host_raw(name, h_ptr, TensorDType::Int32, shape);
}

void TensorDumper::dump_host_int64(
    const std::string& name,
    const int64_t* h_ptr,
    const std::vector<int64_t>& shape
) {
    dump_host_raw(name, h_ptr, TensorDType::Int64, shape);
}