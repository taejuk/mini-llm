#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            printf("CUDA error %s at %s:%d\n",                 \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                          \
        }                                                     \
    } while (0)

__global__ void copy_kernel(
    const float* __restrict__ A,
    float* __restrict__ C,
    size_t N
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += stride) {
        C[i] = A[i];
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    // 256M floats = 1GB per array
    // A + C = 2GB allocation
    size_t N = 256ULL * 1024ULL * 1024ULL;

    float* dA = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

    CUDA_CHECK(cudaMemset(dA, 1, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dC, 0, N * sizeof(float)));

    int block = 256;
    int grid = 4096;

    int warmup = 5;
    int runs = 20;

    for (int i = 0; i < warmup; i++) {
        copy_kernel<<<grid, block>>>(dA, dC, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        copy_kernel<<<grid, block>>>(dA, dC, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double avg_ms = ms / runs;
    double seconds = avg_ms / 1000.0;

    double bytes = 2.0 * N * sizeof(float);
    double gbps = bytes / seconds / 1e9;

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Measured bandwidth: %.1f GB/s\n", gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
