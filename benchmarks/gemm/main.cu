// ceil(a,b) = 올림 나눗셈 — 일부 헤더에서 사용하므로 커스텀 헤더보다 먼저 정의
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


#include "wmma_gemm.cuh"
#include "shared_tensor_core.cuh"
#include "tensor_core_v2_gemm.cuh"

// -----------------------------------------------------------------------------
// Error check
// -----------------------------------------------------------------------------
#define CUDA_CHECK(err)                                                       \
    do {                                                                      \
        cudaError_t err_ = (err);                                             \
        if (err_ != cudaSuccess) {                                            \
            printf("CUDA Error: %s at %s:%d\n",                               \
                   cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(err)                                                     \
    do {                                                                      \
        cublasStatus_t err_ = (err);                                          \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                  \
            printf("cuBLAS Error: %d at %s:%d\n",                             \
                   err_, __FILE__, __LINE__);                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------------------------
// cuBLAS handle
//
// 기존 cublas_gemm.cuh처럼 cublasCreate/cublasDestroy를 GEMM 함수 안에서 매번
// 호출하면 benchmark loop 안에 handle 생성/삭제 overhead가 포함된다.
// 따라서 handle은 main 시작 시 한 번 만들고, 프로그램 끝에서 한 번 해제한다.
// -----------------------------------------------------------------------------
static cublasHandle_t g_cublas_handle = nullptr;

static void cublas_init() {
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));

    // Volta Tensor Core용. FP32 SGEMM에는 사실상 영향이 없고,
    // cublasGemmEx half x half -> float에서 Tensor Core 경로를 활성화한다.
    CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH));
}

static void cublas_destroy() {
    if (g_cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
        g_cublas_handle = nullptr;
    }
}

// -----------------------------------------------------------------------------
// cuBLAS FP32 SGEMM baseline
//
// A, B, C는 row-major.
// cuBLAS는 column-major 기준이므로,
// row-major C = A x B는 column-major 관점에서 C^T = B^T x A^T로 호출한다.
// -----------------------------------------------------------------------------
static void cublas_sgemm_baseline(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
) {
    CUBLAS_CHECK(cublasSgemm(
        g_cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    ));
}

// -----------------------------------------------------------------------------
// cuBLAS Tensor Core baseline
//
// A: half row-major [M x K]
// B: half row-major [K x N]
// C: float row-major [M x N]
//
// 이것도 row-major를 column-major trick으로 호출한다.
// 비교 대상:
//   tensor_core_v2_gemm vs cublas_tensor_core_baseline
// -----------------------------------------------------------------------------
static void cublas_tensor_core_baseline(
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* C
) {
    CUBLAS_CHECK(cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// -----------------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------------
static void rand_init(float* h, int n) {
    for (int i = 0; i < n; i++) {
        h[i] = (float)rand() / RAND_MAX;
    }
}

static void convert_float_to_half(const float* src, half* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

// -----------------------------------------------------------------------------
// Benchmark function types
// -----------------------------------------------------------------------------
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

typedef void (*HalfGemmFn)(
    int,
    int,
    int,
    float,
    const half*,
    const half*,
    float,
    float*
);

// -----------------------------------------------------------------------------
// FP32 benchmark
// -----------------------------------------------------------------------------
static float benchmark_fp32(
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

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    nvtxRangePushA(name);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        fn(M, N, K, alpha, A, B, beta, C);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    nvtxRangePop();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / runs;
}

// -----------------------------------------------------------------------------
// FP16 Tensor Core benchmark
// -----------------------------------------------------------------------------
static float benchmark_half(
    HalfGemmFn fn,
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

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    nvtxRangePushA(name);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        fn(M, N, K, alpha, A, B, beta, C);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    nvtxRangePop();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / runs;
}

// -----------------------------------------------------------------------------
// Print
// -----------------------------------------------------------------------------
static void print_fp32_result(
    const char* name,
    float ms,
    int M,
    int N,
    int K,
    double peak_fp32_gflops,
    float cublas_sgemm_ms
) {
    double flops = 2.0 * M * N * K;
    double bytes = (double)(M * K + K * N + M * N) * sizeof(float);
    double gflops = (flops / 1e9) / (ms / 1e3);
    double ai = flops / bytes;

    printf(
        "%-22s │ %8.3f ms │ %10.1f GFLOPS │ AI=%8.2f │ cuBLAS SGEMM=%6.1f%% │ peak=%6.1f%%\n",
        name,
        ms,
        gflops,
        ai,
        cublas_sgemm_ms / ms * 100.0f,
        gflops / peak_fp32_gflops * 100.0
    );
}

static void print_half_result(
    const char* name,
    float ms,
    int M,
    int N,
    int K,
    float cublas_tc_ms
) {
    double flops = 2.0 * M * N * K;

    // A/B are half, C is float.
    double bytes =
        (double)M * K * sizeof(half) +
        (double)K * N * sizeof(half) +
        (double)M * N * sizeof(float);

    double gflops = (flops / 1e9) / (ms / 1e3);
    double ai = flops / bytes;

    printf(
        "%-22s │ %8.3f ms │ %10.1f GFLOPS │ AI=%8.2f │ cuBLAS TC=%9.1f%%\n",
        name,
        ms,
        gflops,
        ai,
        cublas_tc_ms / ms * 100.0f
    );
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);

    // roofline_bench로 측정한 값
    const double peak_fp32_gflops = 11375.8;
    const double peak_bw_gbs = 606.3;
    const double ridge = peak_fp32_gflops / peak_bw_gbs;

    printf(
        "Peak FP32 GFLOPS: %.1f | Peak BW: %.1f GB/s | Ridge: %.1f FLOP/byte\n",
        peak_fp32_gflops,
        peak_bw_gbs,
        ridge
    );

    printf("──────────────────────────────────────────────────────────────────────────────────────────────\n");

    cublas_init();

    // 큰 size는 naive가 오래 걸린다.
    // 빠르게 보고 싶으면 {1024, 2048, 4096} 정도로 줄이면 된다.
    const int sizes[] = {1024, 2048, 4096, 8192};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    const int warmup = 3;
    const int runs = 10;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si];
        int N = sizes[si];
        int K = sizes[si];

        char size_label[64];
        snprintf(size_label, sizeof(size_label), "size=%d", M);
        nvtxRangePushA(size_label);

        printf("\n[M=N=K=%d]\n", M);

        size_t bytes_A_f32 = (size_t)M * K * sizeof(float);
        size_t bytes_B_f32 = (size_t)K * N * sizeof(float);
        size_t bytes_C_f32 = (size_t)M * N * sizeof(float);

        size_t bytes_A_f16 = (size_t)M * K * sizeof(half);
        size_t bytes_B_f16 = (size_t)K * N * sizeof(half);

        // ---------------------------------------------------------------------
        // Host memory
        // ---------------------------------------------------------------------
        float* hA = (float*)malloc(bytes_A_f32);
        float* hB = (float*)malloc(bytes_B_f32);

        half* hA_half = (half*)malloc(bytes_A_f16);
        half* hB_half = (half*)malloc(bytes_B_f16);

        if (!hA || !hB || !hA_half || !hB_half) {
            printf("Host malloc failed at size %d\n", M);
            exit(1);
        }

        srand(0);
        rand_init(hA, M * K);
        rand_init(hB, K * N);

        convert_float_to_half(hA, hA_half, M * K);
        convert_float_to_half(hB, hB_half, K * N);

        // ---------------------------------------------------------------------
        // Device memory
        // ---------------------------------------------------------------------
        float *dA = nullptr;
        float *dB = nullptr;
        float *dC = nullptr;

        half *dA_half = nullptr;
        half *dB_half = nullptr;

        CUDA_CHECK(cudaMalloc(&dA, bytes_A_f32));
        CUDA_CHECK(cudaMalloc(&dB, bytes_B_f32));
        CUDA_CHECK(cudaMalloc(&dC, bytes_C_f32));

        CUDA_CHECK(cudaMalloc(&dA_half, bytes_A_f16));
        CUDA_CHECK(cudaMalloc(&dB_half, bytes_B_f16));

        CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A_f32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B_f32, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(dA_half, hA_half, bytes_A_f16, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB_half, hB_half, bytes_B_f16, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(dC, 0, bytes_C_f32));

        // ---------------------------------------------------------------------
        // FP32 SGEMM baseline
        // ---------------------------------------------------------------------
        printf("\n[FP32 SGEMM]\n");
        printf(
            "%-22s │ %10s │ %17s │ %11s │ %17s │ %11s\n",
            "Kernel",
            "Time",
            "Throughput",
            "AI",
            "Status",
            "Peak"
        );
        printf("──────────────────────────────────────────────────────────────────────────────────────────────\n");

        float cublas_sgemm_ms = benchmark_fp32(
            cublas_sgemm_baseline,
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
            "cublas_sgemm"
        );

        print_fp32_result(
            "cuBLAS SGEMM",
            cublas_sgemm_ms,
            M,
            N,
            K,
            peak_fp32_gflops,
            cublas_sgemm_ms
        );

        float ms = benchmark_fp32(
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
        print_fp32_result("naive", ms, M, N, K, peak_fp32_gflops, cublas_sgemm_ms);

        ms = benchmark_fp32(
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
        print_fp32_result("coalescing", ms, M, N, K, peak_fp32_gflops, cublas_sgemm_ms);

        ms = benchmark_fp32(
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
        print_fp32_result("caching", ms, M, N, K, peak_fp32_gflops, cublas_sgemm_ms);

        ms = benchmark_fp32(
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
            "blocktiling 1d"
        );
        print_fp32_result("blocktiling 1d", ms, M, N, K, peak_fp32_gflops, cublas_sgemm_ms);

        ms = benchmark_fp32(
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
            "blocktiling 2d"
        );
        print_fp32_result("blocktiling 2d", ms, M, N, K, peak_fp32_gflops, cublas_sgemm_ms);

        ms = benchmark_fp32(
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
        print_fp32_result("vectorized", ms, M, N, K, peak_fp32_gflops, cublas_sgemm_ms);

        // ---------------------------------------------------------------------
        // FP16 Tensor Core baseline
        // ---------------------------------------------------------------------
        printf("\n[FP16 Tensor Core GEMM: half x half -> float]\n");
        printf(
            "%-22s │ %10s │ %17s │ %11s │ %20s\n",
            "Kernel",
            "Time",
            "Throughput",
            "AI",
            "Status"
        );
        printf("──────────────────────────────────────────────────────────────────────────────────────────────\n");

        float cublas_tc_ms = benchmark_half(
            cublas_tensor_core_baseline,
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
            "cublas_tensor_core"
        );

        print_half_result(
            "cuBLAS Tensor Core",
            cublas_tc_ms,
            M,
            N,
            K,
            cublas_tc_ms
        );

        ms = benchmark_half(
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
        print_half_result("wmma tensor core", ms, M, N, K, cublas_tc_ms);

        ms = benchmark_half(
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
        print_half_result("shared tensor core", ms, M, N, K, cublas_tc_ms);

        ms = benchmark_half(
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
        print_half_result("tensor core v2", ms, M, N, K, cublas_tc_ms);

        nvtxRangePop();

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));

        CUDA_CHECK(cudaFree(dA_half));
        CUDA_CHECK(cudaFree(dB_half));

        free(hA);
        free(hB);
        free(hA_half);
        free(hB_half);
    }

    cublas_destroy();

    printf("\n");
    return 0;
}
