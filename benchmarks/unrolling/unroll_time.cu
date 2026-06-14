#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error: %s at %s:%d\n",                             \
                   cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

#define UNROLL_FACTOR 8

__global__ void no_unroll_kernel(float* out, int base_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float x = 1.0f + idx * 0.000001f;

    int total_iters = base_iters * UNROLL_FACTOR;

    // 컴파일러가 자동으로 loop unrolling 하지 못하게 막음
    #pragma unroll 1
    for (int i = 0; i < total_iters; i++) {
        x = __fmaf_rn(x, 1.000001f, 0.000001f);
    }

    out[idx] = x;
}

__global__ void unroll8_kernel(float* out, int base_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float seed = 1.0f + idx * 0.000001f;

    // 서로 독립적인 accumulator 8개
    float x0 = seed + 0.00f;
    float x1 = seed + 0.01f;
    float x2 = seed + 0.02f;
    float x3 = seed + 0.03f;
    float x4 = seed + 0.04f;
    float x5 = seed + 0.05f;
    float x6 = seed + 0.06f;
    float x7 = seed + 0.07f;

    #pragma unroll 1
    for (int i = 0; i < base_iters; i++) {
        // manual loop unrolling x8
        x0 = __fmaf_rn(x0, 1.000001f, 0.000001f);
        x1 = __fmaf_rn(x1, 1.000002f, 0.000002f);
        x2 = __fmaf_rn(x2, 1.000003f, 0.000003f);
        x3 = __fmaf_rn(x3, 1.000004f, 0.000004f);
        x4 = __fmaf_rn(x4, 1.000005f, 0.000005f);
        x5 = __fmaf_rn(x5, 1.000006f, 0.000006f);
        x6 = __fmaf_rn(x6, 1.000007f, 0.000007f);
        x7 = __fmaf_rn(x7, 1.000008f, 0.000008f);
    }

    out[idx] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

template <typename Kernel>
float benchmark(
    Kernel kernel,
    const char* name,
    float* d_out,
    int grid,
    int block,
    int base_iters,
    int runs
) {
    // warm-up
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block>>>(d_out, base_iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        kernel<<<grid, block>>>(d_out, base_iters);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float avg_ms = total_ms / runs;

    printf("%-20s average time: %.3f ms\n", name, avg_ms);

    return avg_ms;
}

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int base_iters = 1 << 20;
    int runs = 10;

    if (argc >= 2) {
        base_iters = atoi(argv[1]);
    }

    if (argc >= 3) {
        runs = atoi(argv[2]);
    }

    int block = 128;

    // 일부러 block 수를 SM 수 정도로 제한해서
    // warp-level parallelism보다 ILP 차이가 잘 보이게 함
    int grid = prop.multiProcessorCount;

    int total_threads = grid * block;

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total_threads * sizeof(float)));

    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Grid: %d, Block: %d\n", grid, block);
    printf("Total threads: %d\n", total_threads);
    printf("Base iterations: %d\n", base_iters);
    printf("Runs: %d\n", runs);
    printf("Unroll factor: %d\n\n", UNROLL_FACTOR);

    float t_no_unroll = benchmark(
        no_unroll_kernel,
        "No unroll",
        d_out,
        grid,
        block,
        base_iters,
        runs
    );

    float t_unroll8 = benchmark(
        unroll8_kernel,
        "Manual unroll x8",
        d_out,
        grid,
        block,
        base_iters,
        runs
    );

    printf("\nSpeedup: %.2fx\n", t_no_unroll / t_unroll8);

    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
