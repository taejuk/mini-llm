// ceil(a,b) = 올림 나눗셈 — 헤더보다 먼저 정의해야 함
#define ceil(a, b) (((a) + (b) - 1) / (b))

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvtx3/nvToolsExt.h>

#include "native_gemm.cuh"
#include "coalescing_gemm.cuh"
#include "caching_gemm.cuh"
#include "blocktiling_1d_gemm.cuh"
#include "blocktiling_2d_gemm.cuh"
#include "vectorized_gemm.cuh"
#include "cublas_gemm.cuh"
#include "wmma_gemm.cuh"
#include "shared_tensor_core.cuh"
#include "tensor_core_v2_gemm.cuh"


static void rand_init(float* h, int n) {
    for (int i = 0; i < n; i++) {
        h[i] = (float)rand() / RAND_MAX;
    }
}

typedef void (*GemmFn)(
    int,
    int,
    int,
    float,
    const float*,
    const float*,
    float,
    float*
);

typedef void (*WmmaGemmFn)(
    int,
    int,
    int,
    float,
    const half*,
    const half*,
    float,
    float*
);

static float benchmark(
    GemmFn fn,
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    int warmup,
    int runs,
    const char* name
) {
    for (int i = 0; i < warmup; i++) {
        fn(M, N, K, alpha, A, B, beta, C);
    }

    cudaDeviceSynchronize();

    nvtxRangePushA(name);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < runs; i++) {
        fn(M, N, K, alpha, A, B, beta, C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    nvtxRangePop();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / runs;
}

static float benchmark_wmma(
    WmmaGemmFn fn,
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C,
    int warmup,
    int runs,
    const char* name
) {
    for (int i = 0; i < warmup; i++) {
        fn(M, N, K, alpha, A, B, beta, C);
    }

    cudaDeviceSynchronize();

    nvtxRangePushA(name);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < runs; i++) {
        fn(M, N, K, alpha, A, B, beta, C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    nvtxRangePop();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / runs;
}

static void print_result(
    const char* name,
    float ms,
    int M,
    int N,
    int K,
    double peak_gflops,
    double peak_bw_gbs,
    float standard
) {
    double flops = 2.0 * M * N * K;
    double bytes = static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    double gflops = (flops / 1e9) / (ms / 1e3);

    printf(
        "%-20s │ %6.2f ms │ %8.1f GFLOPS │ cublas=%.0f%% (%.0f%%)\n",
        name,
        ms,
        gflops,
        standard / ms * 100.0,
        gflops / peak_gflops * 100.0
    );
}

static void print_wmma_result(
    const char* name,
    float ms,
    int M,
    int N,
    int K,
    float standard
) {
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1e3);

    double bytes =
        static_cast<double>(M) * K * sizeof(half) +
        static_cast<double>(K) * N * sizeof(half) +
        static_cast<double>(M) * N * sizeof(float);

    double ai = flops / bytes;

    printf(
        "%-20s │ %6.2f ms │ %8.1f GFLOPS │ AI=%6.2f │ cublas_sgemm=%.0f%%\n",
        name,
        ms,
        gflops,
        ai,
        standard / ms * 100.0f
    );
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n", prop.name);

    double peak_gflops = 11375.8;
    double peak_bw_gbs = 606.3;
    double ridge = peak_gflops / peak_bw_gbs;

    printf(
        "Peak GFLOPS: %.1f | Peak BW: %.1f GB/s | Ridge: %.1f FLOP/byte\n",
        peak_gflops,
        peak_bw_gbs,
        ridge
    );

    printf("─────────────────────────────────────────────────────────────\n");

    const int sizes[] = {1024,2048,4096,8192};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si];
        int N = sizes[si];
        int K = sizes[si];

        char size_label[32];
        snprintf(size_label, sizeof(size_label), "size=%d", sizes[si]);
        nvtxRangePushA(size_label);

        printf("\n[M=N=K=%d]\n", M);
        printf(
            "%-20s │ %8s │ %18s │ %10s │ %s\n",
            "Kernel",
            "Time",
            "Throughput",
            "AI",
            "Status"
        );
        printf("─────────────────────────────────────────────────────────────\n");

        float* hA = (float*)malloc(M * K * sizeof(float));
        float* hB = (float*)malloc(K * N * sizeof(float));
        float* hC = (float*)malloc(M * N * sizeof(float));

        rand_init(hA, M * K);
        rand_init(hB, K * N);

        float *dA, *dB, *dC;
        cudaMalloc(&dA, M * K * sizeof(float));
        cudaMalloc(&dB, K * N * sizeof(float));
        cudaMalloc(&dC, M * N * sizeof(float));

        cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);

        half* hA_half = (half*)malloc(M * K * sizeof(half));
        half* hB_half = (half*)malloc(K * N * sizeof(half));

        for (int i = 0; i < M * K; i++) {
            hA_half[i] = __float2half(hA[i]);
        }

        for (int i = 0; i < K * N; i++) {
            hB_half[i] = __float2half(hB[i]);
        }

        half *dA_half, *dB_half;
        cudaMalloc(&dA_half, M * K * sizeof(half));
        cudaMalloc(&dB_half, K * N * sizeof(half));

        cudaMemcpy(
            dA_half,
            hA_half,
            M * K * sizeof(half),
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            dB_half,
            hB_half,
            K * N * sizeof(half),
            cudaMemcpyHostToDevice
        );

        int warmup = 3;
        int runs = 10;

        float alpha = 1.0f;
        float beta = 0.0f;

        float standard = benchmark(
            cublas_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "cublas"
        );
        
        float ms = benchmark(
            naive_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "naive"
        );
        print_result("naive", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        ms = benchmark(
            coalescing_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "coalescing"
        );
        print_result("coalescing", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        ms = benchmark(
            caching_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "caching"
        );
        print_result("caching", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        ms = benchmark(
            blocktiling_1d_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "block tiling 1d"
        );
        print_result("blocktiling 1d", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        ms = benchmark(
            blocktiling_2d_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "block tiling 2d"
        );
        print_result("blocktiling 2d", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        ms = benchmark(
            vectorized_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            warmup,
            runs,
            "vectorized"
        );
        print_result("vectorized", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        ms = benchmark_wmma(
            wmma_gemm,
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC,
            warmup,
            runs,
            "wmma tensor core"
        );

        print_wmma_result(
            "wmma tensor core",
            ms,
            M,
            N,
            K,
            standard
        );
        ms = benchmark_wmma(
            shared_tensor_core_gemm,
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC,
            warmup,
            runs,
            "shared tensor core"
        );

        print_wmma_result(
            "shared tensor core",
            ms,
            M,
            N,
    		K,
             standard
        );

        ms = benchmark_wmma(
            tensor_core_v2_gemm,
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC,
            warmup,
            runs,
            "tensor core v2"
        );

        print_wmma_result(
            "tensor core v2",
            ms,
            M,
            N,
            K,
            standard
        );

        nvtxRangePop();

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaFree(dA_half);
        cudaFree(dB_half);

        free(hA);
        free(hB);
        free(hC);

        free(hA_half);
        free(hB_half);
    }

    printf("\n");
    return 0;
}
