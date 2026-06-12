#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            printf("CUDA error: %s at %s:%d\n",                             \
                   cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// One loop iteration performs 32 FP32 FMA operations.
// One FMA = 1 multiply + 1 add = 2 FLOPs.
#define FMA_PER_ITER 32

__global__ void fp32_peak_kernel(
    float* __restrict__ out,
    int repeat
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float seed = 1.0f + (idx & 1023) * 0.000001f;

    // Independent accumulator chains.
    // Multiple independent accumulators help hide FMA latency and expose
    // more instruction-level parallelism.
    float x00 = seed + 0.00f;
    float x01 = seed + 0.01f;
    float x02 = seed + 0.02f;
    float x03 = seed + 0.03f;
    float x04 = seed + 0.04f;
    float x05 = seed + 0.05f;
    float x06 = seed + 0.06f;
    float x07 = seed + 0.07f;

    float x08 = seed + 0.08f;
    float x09 = seed + 0.09f;
    float x10 = seed + 0.10f;
    float x11 = seed + 0.11f;
    float x12 = seed + 0.12f;
    float x13 = seed + 0.13f;
    float x14 = seed + 0.14f;
    float x15 = seed + 0.15f;

    float x16 = seed + 0.16f;
    float x17 = seed + 0.17f;
    float x18 = seed + 0.18f;
    float x19 = seed + 0.19f;
    float x20 = seed + 0.20f;
    float x21 = seed + 0.21f;
    float x22 = seed + 0.22f;
    float x23 = seed + 0.23f;

    float x24 = seed + 0.24f;
    float x25 = seed + 0.25f;
    float x26 = seed + 0.26f;
    float x27 = seed + 0.27f;
    float x28 = seed + 0.28f;
    float x29 = seed + 0.29f;
    float x30 = seed + 0.30f;
    float x31 = seed + 0.31f;

    // Do not unroll the outer loop completely.
    // The loop count is intentionally runtime-controlled.
    #pragma unroll 1
    for (int r = 0; r < repeat; r++) {
        x00 = __fmaf_rn(x00, 1.000001f, 0.000001f);
        x01 = __fmaf_rn(x01, 1.000002f, 0.000002f);
        x02 = __fmaf_rn(x02, 1.000003f, 0.000003f);
        x03 = __fmaf_rn(x03, 1.000004f, 0.000004f);

        x04 = __fmaf_rn(x04, 1.000005f, 0.000005f);
        x05 = __fmaf_rn(x05, 1.000006f, 0.000006f);
        x06 = __fmaf_rn(x06, 1.000007f, 0.000007f);
        x07 = __fmaf_rn(x07, 1.000008f, 0.000008f);

        x08 = __fmaf_rn(x08, 1.000009f, 0.000009f);
        x09 = __fmaf_rn(x09, 1.000010f, 0.000010f);
        x10 = __fmaf_rn(x10, 1.000011f, 0.000011f);
        x11 = __fmaf_rn(x11, 1.000012f, 0.000012f);

        x12 = __fmaf_rn(x12, 1.000013f, 0.000013f);
        x13 = __fmaf_rn(x13, 1.000014f, 0.000014f);
        x14 = __fmaf_rn(x14, 1.000015f, 0.000015f);
        x15 = __fmaf_rn(x15, 1.000016f, 0.000016f);

        x16 = __fmaf_rn(x16, 1.000017f, 0.000017f);
        x17 = __fmaf_rn(x17, 1.000018f, 0.000018f);
        x18 = __fmaf_rn(x18, 1.000019f, 0.000019f);
        x19 = __fmaf_rn(x19, 1.000020f, 0.000020f);

        x20 = __fmaf_rn(x20, 1.000021f, 0.000021f);
        x21 = __fmaf_rn(x21, 1.000022f, 0.000022f);
        x22 = __fmaf_rn(x22, 1.000023f, 0.000023f);
        x23 = __fmaf_rn(x23, 1.000024f, 0.000024f);

        x24 = __fmaf_rn(x24, 1.000025f, 0.000025f);
        x25 = __fmaf_rn(x25, 1.000026f, 0.000026f);
        x26 = __fmaf_rn(x26, 1.000027f, 0.000027f);
        x27 = __fmaf_rn(x27, 1.000028f, 0.000028f);

        x28 = __fmaf_rn(x28, 1.000029f, 0.000029f);
        x29 = __fmaf_rn(x29, 1.000030f, 0.000030f);
        x30 = __fmaf_rn(x30, 1.000031f, 0.000031f);
        x31 = __fmaf_rn(x31, 1.000032f, 0.000032f);
    }

    // Store all accumulators so the compiler cannot remove the computation.
    out[idx] =
        x00 + x01 + x02 + x03 +
        x04 + x05 + x06 + x07 +
        x08 + x09 + x10 + x11 +
        x12 + x13 + x14 + x15 +
        x16 + x17 + x18 + x19 +
        x20 + x21 + x22 + x23 +
        x24 + x25 + x26 + x27 +
        x28 + x29 + x30 + x31;
}

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int repeat = 8192;
    int runs = 20;
    int warmup = 5;

    // More blocks than SMs to keep the GPU saturated.
    int block = 256;
    int grid_multiplier = 32;
    int grid = prop.multiProcessorCount * grid_multiplier;

    if (argc >= 2) {
        repeat = atoi(argv[1]);
    }

    if (argc >= 3) {
        runs = atoi(argv[2]);
    }

    if (argc >= 4) {
        grid_multiplier = atoi(argv[3]);
        grid = prop.multiProcessorCount * grid_multiplier;
    }

    int num_threads = block * grid;

    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Block size: %d\n", block);
    printf("Grid size: %d\n", grid);
    printf("Total threads: %d\n", num_threads);
    printf("Repeat: %d\n", repeat);
    printf("Runs: %d\n", runs);
    printf("FMA per iteration per thread: %d\n", FMA_PER_ITER);

    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)num_threads * sizeof(float)));

    for (int i = 0; i < warmup; i++) {
        fp32_peak_kernel<<<grid, block>>>(d_out, repeat);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        fp32_peak_kernel<<<grid, block>>>(d_out, repeat);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    double avg_ms = total_ms / runs;
    double avg_seconds = avg_ms / 1000.0;

    // Per kernel:
    //   num_threads * repeat * FMA_PER_ITER FMA instructions
    //
    // 1 FMA = 2 FLOPs.
    double flops =
        (double)num_threads *
        (double)repeat *
        (double)FMA_PER_ITER *
        2.0;

    double gflops = flops / avg_seconds / 1e9;

    printf("\n");
    printf("Average time: %.3f ms\n", avg_ms);
    printf("FLOPs per kernel: %.3e\n", flops);
    printf("Measured FP32 peak: %.1f GFLOPS\n", gflops);
    printf("Measured FP32 peak: %.3f TFLOPS\n", gflops / 1000.0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
