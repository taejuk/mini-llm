// ceil(a,b) = 올림 나눗셈 — 헤더보다 먼저 정의해야 함
#define ceil(a, b) (((a) + (b) - 1) / (b))

#include <cuda_runtime.h>
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

// ── 행렬 초기화 ────────────────────────────────────────────────
static void rand_init(float* h, int n) {
    for (int i = 0; i < n; i++)
        h[i] = (float)rand() / RAND_MAX;
}

// ── 커널 벤치마크 ──────────────────────────────────────────────
typedef void (*GemmFn)(int, int, int, float,const float*,const float*, float,float*);

static float benchmark(GemmFn fn,
                       int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C,
                       int warmup, int runs, const char* name)
{
    // 워밍업
    for (int i = 0; i < warmup; i++)
        fn(M, N, K, alpha ,A, B, beta,C);
    cudaDeviceSynchronize();
    
    nvtxRangePushA(name);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++)
        fn(M, N, K, alpha ,A, B, beta,C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    nvtxRangePop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

// ── 결과 출력 ──────────────────────────────────────────────────
static void print_result(const char* name,
                         float ms, int M, int N, int K,
                         double peak_gflops, double peak_bw_gbs, float standard)
{
    double flops   = 2.0 * M * N * K;
    double bytes   = (double)(M*K + K*N + M*N) * sizeof(float);
    double gflops  = (flops / 1e9) / (ms / 1e3);
    double ai      = flops / bytes;
    double ridge   = (peak_gflops * 1e9) / (peak_bw_gbs * 1e9);

//    const char* bound = (ai < ridge) ? "Memory Bound" : "Compute Bound";

    printf("%-20s │ %6.2f ms │ %8.1f GFLOPS │ cublas=%.0f%% (%.0f%%)\n",
           name, ms, gflops, standard / ms * 100, gflops / peak_gflops * 100.0);
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
    const int sizes[] = {4096};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si], N = sizes[si], K = sizes[si];
        char size_label[32];
	snprintf(size_label, sizeof(size_label), "size=%d", sizes[si]);
	nvtxRangePushA(size_label);
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
        float alpha = 1.0f; float beta = 0.0f;
        float standard = benchmark(cublas_gemm, M, N, K, alpha, dA, dB, beta, dC, warmup, runs, "cublas");
	// naive
        float ms = benchmark(naive_gemm, M, N, K, alpha, dA, dB, beta, dC, warmup, runs,"naive");
        print_result("naive", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        // coalescing
        ms = benchmark(coalescing_gemm, M, N, K, alpha, dA, dB, beta, dC, warmup, runs, "coalescing");
        print_result("coalescing", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);

        // caching
        ms = benchmark(caching_gemm, M, N, K, alpha, dA, dB, beta, dC, warmup, runs, "caching");
        print_result("caching", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);
	
	ms = benchmark(blocktiling_1d_gemm, M, N, K, alpha, dA, dB, beta, dC, warmup, runs, "block tilind 1d");
	print_result("blocktiling 1d", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);
	
	ms = benchmark(blocktiling_2d_gemm, M, N, K, alpha, dA, dB, beta, dC, warmup, runs, "block tiling 2d");
        print_result("blocktiling 2d", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);
	
	ms = benchmark(vectorized_gemm, M, N, K, alpha, dA, dB, beta,dC, warmup, runs, "vectorized");
	print_result("vectorized", ms, M, N, K, peak_gflops, peak_bw_gbs, standard);
	nvtxRangePop();

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB); free(hC);
    }

    printf("\n");
    return 0;
}
