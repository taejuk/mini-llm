// ceil(a,b) = 올림 나눗셈 — 헤더보다 먼저 정의해야 함
#define ceil(a, b) (((a) + (b) - 1) / (b))

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "native_gemm.cuh"
#include "coalescing_gemm.cuh"
#include "caching_gemm.cuh"

// ── 행렬 초기화 ────────────────────────────────────────────────
static void rand_init(float* h, int n) {
    for (int i = 0; i < n; i++)
        h[i] = (float)rand() / RAND_MAX;
}

// ── 커널 벤치마크 ──────────────────────────────────────────────
typedef void (*GemmFn)(float*, float*, float*, int, int, int);

static float benchmark(GemmFn fn,
                       float* dA, float* dB, float* dC,
                       int M, int N, int K,
                       int warmup, int runs)
{
    // 워밍업
    for (int i = 0; i < warmup; i++)
        fn(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
        fn(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

// ── 결과 출력 ──────────────────────────────────────────────────
static void print_result(const char* name,
                         float ms, int M, int N, int K,
                         double peak_gflops, double peak_bw_gbs)
{
    double flops   = 2.0 * M * N * K;
    double bytes   = (double)(M*K + K*N + M*N) * sizeof(float);
    double gflops  = (flops / 1e9) / (ms / 1e3);
    double ai      = flops / bytes;
    double ridge   = (peak_gflops * 1e9) / (peak_bw_gbs * 1e9);

//    const char* bound = (ai < ridge) ? "Memory Bound" : "Compute Bound";

    printf("%-20s │ %6.2f ms │ %8.1f GFLOPS │ AI=%6.2f (%.0f%%)\n",
           name, ms, gflops, ai, gflops / peak_gflops * 100.0);
}

int main() {
    // GPU 정보
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    // Roofline 수치 (roofline_bench로 측정한 값)
    double peak_gflops = 11375.8;
    double peak_bw_gbs = 606.3;
    double ridge = peak_gflops / peak_bw_gbs;
    printf("Peak GFLOPS: %.1f | Peak BW: %.1f GB/s | Ridge: %.1f FLOP/byte\n",
           peak_gflops, peak_bw_gbs, ridge);
    printf("─────────────────────────────────────────────────────────────\n");

    // 행렬 크기 (M=N=K)
    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];

        printf("\n[M=N=K=%d]\n", M);
        printf("%-20s │ %8s │ %18s │ %10s │ %s\n",
               "Kernel", "Time", "Throughput", "AI", "Status");
        printf("─────────────────────────────────────────────────────────────\n");

        // 호스트 메모리
        float* hA = (float*)malloc(M * K * sizeof(float));
        float* hB = (float*)malloc(K * N * sizeof(float));
        float* hC = (float*)malloc(M * N * sizeof(float));
        rand_init(hA, M * K);
        rand_init(hB, K * N);

        // 디바이스 메모리
        float *dA, *dB, *dC;
        cudaMalloc(&dA, M * K * sizeof(float));
        cudaMalloc(&dB, K * N * sizeof(float));
        cudaMalloc(&dC, M * N * sizeof(float));
        cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);

        int warmup = 3, runs = 10;

        // naive
        float ms = benchmark(naive_gemm, dA, dB, dC, M, N, K, warmup, runs);
        print_result("naive", ms, M, N, K, peak_gflops, peak_bw_gbs);

        // coalescing
        ms = benchmark(coalescing_gemm, dA, dB, dC, M, N, K, warmup, runs);
        print_result("coalescing", ms, M, N, K, peak_gflops, peak_bw_gbs);

        // caching
        ms = benchmark(caching_gemm, dA, dB, dC, M, N, K, warmup, runs);
        print_result("caching", ms, M, N, K, peak_gflops, peak_bw_gbs);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB); free(hC);
    }

    printf("\n");
    return 0;
}
