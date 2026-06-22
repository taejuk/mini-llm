// ceil(a,b) = 올림 나눗셈 — 일부 헤더에서 사용하므로 커스텀 헤더보다 먼저 정의
#define ceil(a, b) (((a) + (b) - 1) / (b))

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <nvtx3/nvToolsExt.h>

#include "native_gemm.cuh"
#include "coalescing_gemm.cuh"
#include "caching_gemm.cuh"
#include "blocktiling_1d_gemm.cuh"
#include "blocktiling_2d_gemm.cuh"
#include "vectorized_gemm.cuh"
#include "vectorized_bank_gemm.cuh"
#include "vectorized_column_major_gemm.cuh"

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
// -----------------------------------------------------------------------------
static cublasHandle_t g_cublas_handle = nullptr;

static void cublas_init() {
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));

    // Volta Tensor Core용.
    // FP32 SGEMM에는 큰 영향이 없고,
    // cublasGemmEx half x half -> float에서 Tensor Core 경로를 활성화한다.
    //
    // 필요하면 활성화:
    // CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH));
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
// row-major C = A x B를 column-major trick으로 호출한다.
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
// Correctness check
// -----------------------------------------------------------------------------
static bool compare_device_outputs(
    const char* name,
    const float* d_ref,
    const float* d_out,
    float* h_ref,
    float* h_out,
    int M,
    int N,
    float atol,
    float rtol
) {
    const size_t total = (size_t)M * (size_t)N;
    const size_t bytes = total * sizeof(float);

    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    size_t bad_count = 0;
    size_t bad_idx = (size_t)-1;

    for (size_t i = 0; i < total; i++) {
        float ref = h_ref[i];
        float out = h_out[i];

        float abs_err = fabsf(ref - out);
        float rel_err = abs_err / fmaxf(fabsf(ref), 1e-6f);
        float tol = atol + rtol * fabsf(ref);

        sum_abs += abs_err;

        if (abs_err > max_abs) {
            max_abs = abs_err;
        }

        if (rel_err > max_rel) {
            max_rel = rel_err;
        }

        if (!isfinite(out) || abs_err > tol) {
            bad_count++;

            if (bad_idx == (size_t)-1) {
                bad_idx = i;
            }
        }
    }

    double mean_abs = sum_abs / (double)total;
    bool pass = bad_count == 0;

    printf(
        "  [check] %-26s │ %s │ max_abs=%10.6f │ mean_abs=%10.6f │ max_rel=%10.6f │ bad=%zu/%zu",
        name,
        pass ? "PASS" : "FAIL",
        max_abs,
        mean_abs,
        max_rel,
        bad_count,
        total
    );

    if (!pass && bad_idx != (size_t)-1) {
        printf(
            " │ first_bad_idx=%zu ref=%f out=%f",
            bad_idx,
            h_ref[bad_idx],
            h_out[bad_idx]
        );
    }

    printf("\n");

    return pass;
}

static bool verify_fp32_kernel(
    const char* name,
    GemmFn fn,
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* dC_test,
    const float* dC_ref,
    float* hC_ref,
    float* hC_test,
    float atol,
    float rtol
) {
    size_t bytes_C = (size_t)M * (size_t)N * sizeof(float);

    CUDA_CHECK(cudaMemset(dC_test, 0, bytes_C));

    fn(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        dC_test
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return compare_device_outputs(
        name,
        dC_ref,
        dC_test,
        hC_ref,
        hC_test,
        M,
        N,
        atol,
        rtol
    );
}

static bool verify_half_kernel(
    const char* name,
    HalfGemmFn fn,
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* dC_test,
    const float* dC_ref,
    float* hC_ref,
    float* hC_test,
    float atol,
    float rtol
) {
    size_t bytes_C = (size_t)M * (size_t)N * sizeof(float);

    CUDA_CHECK(cudaMemset(dC_test, 0, bytes_C));

    fn(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        dC_test
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return compare_device_outputs(
        name,
        dC_ref,
        dC_test,
        hC_ref,
        hC_test,
        M,
        N,
        atol,
        rtol
    );
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
        "%-26s │ %8.3f ms │ %10.1f GFLOPS │ AI=%8.2f │ cuBLAS SGEMM=%6.1f%% │ peak=%6.1f%%\n",
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
        "%-26s │ %8.3f ms │ %10.1f GFLOPS │ AI=%8.2f │ cuBLAS TC=%9.1f%%\n",
        name,
        ms,
        gflops,
        ai,
        cublas_tc_ms / ms * 100.0f
    );
}

// -----------------------------------------------------------------------------
// Run one kernel: benchmark + correctness
// -----------------------------------------------------------------------------
static bool run_fp32_kernel(
    const char* name,
    const char* nvtx_name,
    GemmFn fn,
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* dC_bench,
    float* dC_test,
    const float* dC_ref,
    float* hC_ref,
    float* hC_test,
    int warmup,
    int runs,
    double peak_fp32_gflops,
    float cublas_sgemm_ms,
    float atol,
    float rtol
) {
    float ms = benchmark_fp32(
        fn,
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        dC_bench,
        warmup,
        runs,
        nvtx_name
    );

    print_fp32_result(
        name,
        ms,
        M,
        N,
        K,
        peak_fp32_gflops,
        cublas_sgemm_ms
    );

    return verify_fp32_kernel(
        name,
        fn,
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        dC_test,
        dC_ref,
        hC_ref,
        hC_test,
        atol,
        rtol
    );
}

static bool run_half_kernel(
    const char* name,
    const char* nvtx_name,
    HalfGemmFn fn,
    int M,
    int N,
    int K,
    float alpha,
    const half* A,
    const half* B,
    float beta,
    float* dC_bench,
    float* dC_test,
    const float* dC_ref,
    float* hC_ref,
    float* hC_test,
    int warmup,
    int runs,
    float cublas_tc_ms,
    float atol,
    float rtol
) {
    float ms = benchmark_half(
        fn,
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        dC_bench,
        warmup,
        runs,
        nvtx_name
    );

    print_half_result(
        name,
        ms,
        M,
        N,
        K,
        cublas_tc_ms
    );

    return verify_half_kernel(
        name,
        fn,
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        dC_test,
        dC_ref,
        hC_ref,
        hC_test,
        atol,
        rtol
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
    // 빠르게 보고 싶으면 {1024, 2048} 정도로 줄이면 된다.
    const int sizes[] = {1024, 2048, 4096};
    const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    const int warmup = 3;
    const int runs = 10;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // FP32 custom kernel은 cuBLAS SGEMM과 비교.
    // accumulation order가 다르므로 bitwise match는 기대하지 않는다.
    const float fp32_atol = 1e-2f;
    const float fp32_rtol = 1e-4f;

    // FP16 Tensor Core는 half multiply + float accumulate이고,
    // kernel마다 accumulation order가 다르므로 tolerance를 더 크게 둔다.
    const float half_atol = 2.0f;
    const float half_rtol = 2e-2f;

    bool all_checks_pass = true;

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

        float* hC_ref = (float*)malloc(bytes_C_f32);
        float* hC_test = (float*)malloc(bytes_C_f32);

        half* hA_half = (half*)malloc(bytes_A_f16);
        half* hB_half = (half*)malloc(bytes_B_f16);

        if (!hA || !hB || !hC_ref || !hC_test || !hA_half || !hB_half) {
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
        float *dAT = nullptr;
        float *dB = nullptr;
        float *dC = nullptr;
        float *dC_ref = nullptr;
        float *dC_test = nullptr;

        half *dA_half = nullptr;
        half *dB_half = nullptr;

        CUDA_CHECK(cudaMalloc(&dA, bytes_A_f32));
        CUDA_CHECK(cudaMalloc(&dAT, bytes_A_f32));
        CUDA_CHECK(cudaMalloc(&dB, bytes_B_f32));
        CUDA_CHECK(cudaMalloc(&dC, bytes_C_f32));
        CUDA_CHECK(cudaMalloc(&dC_ref, bytes_C_f32));
        CUDA_CHECK(cudaMalloc(&dC_test, bytes_C_f32));

        CUDA_CHECK(cudaMalloc(&dA_half, bytes_A_f16));
        CUDA_CHECK(cudaMalloc(&dB_half, bytes_B_f16));

        CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A_f32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B_f32, cudaMemcpyHostToDevice));

        transpose_A_for_vectorized_column_major_a(M, K, dA, dAT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(dA_half, hA_half, bytes_A_f16, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB_half, hB_half, bytes_B_f16, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(dC, 0, bytes_C_f32));
        CUDA_CHECK(cudaMemset(dC_ref, 0, bytes_C_f32));
        CUDA_CHECK(cudaMemset(dC_test, 0, bytes_C_f32));

        // ---------------------------------------------------------------------
        // FP32 correctness reference
        // ---------------------------------------------------------------------
        cublas_sgemm_baseline(
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC_ref
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------------------------------------------------------------------
        // FP32 SGEMM benchmark
        // ---------------------------------------------------------------------
        printf("\n[FP32 SGEMM]\n");
        printf(
            "%-26s │ %10s │ %17s │ %11s │ %17s │ %11s\n",
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

        // cuBLAS baseline 자체도 ref와 동일한지 확인하고 싶으면 아래를 사용할 수 있다.
        // 여기서는 reference가 cuBLAS이므로 custom kernel들만 검사한다.

        all_checks_pass &= run_fp32_kernel(
            "naive",
            "naive",
            naive_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        all_checks_pass &= run_fp32_kernel(
            "coalescing",
            "coalescing",
            coalescing_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        all_checks_pass &= run_fp32_kernel(
            "caching",
            "caching",
            caching_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        all_checks_pass &= run_fp32_kernel(
            "blocktiling 1d",
            "blocktiling_1d",
            blocktiling_1d_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        all_checks_pass &= run_fp32_kernel(
            "blocktiling 2d",
            "blocktiling_2d",
            blocktiling_2d_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        all_checks_pass &= run_fp32_kernel(
            "vectorized",
            "vectorized",
            vectorized_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        all_checks_pass &= run_fp32_kernel(
            "vectorized bank conflict",
            "vectorized_bank",
            vectorized_bank_gemm,
            M,
            N,
            K,
            alpha,
            dA,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        // vectorized_column_major_gemm은 A를 transpose한 dAT를 입력으로 받는다.
        all_checks_pass &= run_fp32_kernel(
            "vectorized column major",
            "vectorized_column_major",
            vectorized_column_major_gemm,
            M,
            N,
            K,
            alpha,
            dAT,
            dB,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            peak_fp32_gflops,
            cublas_sgemm_ms,
            fp32_atol,
            fp32_rtol
        );

        // ---------------------------------------------------------------------
        // FP16 Tensor Core correctness reference
        // ---------------------------------------------------------------------
        CUDA_CHECK(cudaMemset(dC_ref, 0, bytes_C_f32));

        cublas_tensor_core_baseline(
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC_ref
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---------------------------------------------------------------------
        // FP16 Tensor Core baseline
        // ---------------------------------------------------------------------
        printf("\n[FP16 Tensor Core GEMM: half x half -> float]\n");
        printf(
            "%-26s │ %10s │ %17s │ %11s │ %20s\n",
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

        all_checks_pass &= run_half_kernel(
            "wmma tensor core",
            "wmma_tensor_core",
            wmma_gemm,
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            cublas_tc_ms,
            half_atol,
            half_rtol
        );

        all_checks_pass &= run_half_kernel(
            "shared tensor core",
            "shared_tensor_core",
            shared_tensor_core_gemm,
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            cublas_tc_ms,
            half_atol,
            half_rtol
        );

        all_checks_pass &= run_half_kernel(
            "tensor core v2",
            "tensor_core_v2",
            tensor_core_v2_gemm,
            M,
            N,
            K,
            alpha,
            dA_half,
            dB_half,
            beta,
            dC,
            dC_test,
            dC_ref,
            hC_ref,
            hC_test,
            warmup,
            runs,
            cublas_tc_ms,
            half_atol,
            half_rtol
        );

        nvtxRangePop();

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dAT));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dC_ref));
        CUDA_CHECK(cudaFree(dC_test));

        CUDA_CHECK(cudaFree(dA_half));
        CUDA_CHECK(cudaFree(dB_half));

        free(hA);
        free(hB);
        free(hC_ref);
        free(hC_test);
        free(hA_half);
        free(hB_half);
    }

    cublas_destroy();

    printf("\n");
    printf("Correctness summary: %s\n", all_checks_pass ? "PASS" : "FAIL");

    return all_checks_pass ? 0 : 1;
}