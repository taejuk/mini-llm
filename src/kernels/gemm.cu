#include "kernels/gemm.cuh"
#include "kernels/decode_gemm.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <cstdlib>
#include <iostream>

namespace mini_llm::kernels {

namespace {

bool use_cublas_gemm() {
    static bool enabled =
        std::getenv("MINI_LLM_USE_CUBLAS") != nullptr;
    return enabled;
}

void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr
            << "[cuBLAS] "
            << what
            << " failed with status "
            << static_cast<int>(status)
            << "\n";
        std::exit(1);
    }
}

cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        check_cublas(
            cublasCreate(&h),
            "cublasCreate"
        );
        return h;
    }();

    return handle;
}

__global__ void add_bias_kernel(
    float* C,
    const float* bias,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) {
        return;
    }

    int col = idx % N;
    C[idx] += bias[col];
}

void launch_add_bias(
    float* C,
    const float* bias,
    int M,
    int N,
    cudaStream_t stream
) {
    if (bias == nullptr) {
        return;
    }

    int total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;

    add_bias_kernel<<<grid, block, 0, stream>>>(
        C,
        bias,
        M,
        N
    );
}

/*
    Existing mini-llm layout:
        A: row-major [M, K]
        B: row-major [K, N]
        C: row-major [M, N]

    cuBLAS assumes column-major.

    Row-major C = A * B is equivalent to:
        C_col = B_col * A_col

    So we call cublasSgemm as:
        m = N
        n = M
        k = K
        A = B, lda = N
        B = A, ldb = K
        C = C, ldc = N
*/
void launch_gemm_cublas_row_major(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    cudaStream_t stream
) {
    cublasHandle_t handle = get_cublas_handle();

    check_cublas(
        cublasSetStream(handle, stream),
        "cublasSetStream"
    );

    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B,
            N,
            A,
            K,
            &beta,
            C,
            N
        ),
        "cublasSgemm row-major"
    );
}

bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

bool can_use_vectorized_bank_gemm(
    int M,
    int N,
    int K,
    const float* A,
    const float* B,
    const float* C
) {
    if ((M % BM) != 0 || (N % BN) != 0 || (K % BK) != 0) {
        return false;
    }

    if (!is_aligned_16(A) || !is_aligned_16(B) || !is_aligned_16(C)) {
        return false;
    }

    return true;
}

constexpr int SA_STRIDE = BM + 4;
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

__device__ __forceinline__ float4 make_zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float4 load_float4_or_scalar(
    const float* __restrict__ ptr,
    int row,
    int col,
    int ld,
    int rows,
    int cols
) {
    float4 tmp = make_zero_float4();

    if (row >= rows) {
        return tmp;
    }

    size_t base =
        static_cast<size_t>(row) *
        static_cast<size_t>(ld) +
        static_cast<size_t>(col);

    const float* addr = ptr + base;

    bool in_bounds_vec =
        (col + 3 < cols);

    bool aligned_16 =
        (reinterpret_cast<uintptr_t>(addr) & 0xF) == 0;

    if (in_bounds_vec && aligned_16) {
        return reinterpret_cast<const float4*>(addr)[0];
    }

    if (col + 0 < cols) {
        tmp.x = ptr[base + 0];
    }

    if (col + 1 < cols) {
        tmp.y = ptr[base + 1];
    }

    if (col + 2 < cols) {
        tmp.z = ptr[base + 2];
    }

    if (col + 3 < cols) {
        tmp.w = ptr[base + 3];
    }

    return tmp;
}

__global__ void gemm_general_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    const float* __restrict__ bias,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float sA[BK * SA_STRIDE];
    __shared__ float sB[BK * BN];

    const int blockRow = cRow * BM;
    const int blockCol = cCol * BN;

    float threadResults[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        const int numAFloat4 = (BM * BK) / 4;

        for (int i = threadIdx.x; i < numAFloat4; i += NUM_THREADS) {
            int rowA = i / (BK / 4);
            int colA4 = i % (BK / 4);

            int globalRowA = blockRow + rowA;
            int globalColA = bkIdx + colA4 * 4;

            float4 tmp = load_float4_or_scalar(
                A,
                globalRowA,
                globalColA,
                K,
                M,
                K
            );

            sA[(colA4 * 4 + 0) * SA_STRIDE + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * SA_STRIDE + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * SA_STRIDE + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * SA_STRIDE + rowA] = tmp.w;
        }

        const int numBFloat4 = (BK * BN) / 4;

        for (int i = threadIdx.x; i < numBFloat4; i += NUM_THREADS) {
            int rowB = i / (BN / 4);
            int colB4 = i % (BN / 4);

            int globalRowB = bkIdx + rowB;
            int globalColB = blockCol + colB4 * 4;

            float4 tmp = load_float4_or_scalar(
                B,
                globalRowB,
                globalColB,
                N,
                K,
                N
            );

            reinterpret_cast<float4*>(
                &sB[rowB * BN + colB4 * 4]
            )[0] = tmp;
        }

        __syncthreads();

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            const int aOffset =
                dotIdx * SA_STRIDE + threadRow * TM;

            const int bOffset =
                dotIdx * BN + threadCol * TN;

            float4 a0 = reinterpret_cast<float4*>(
                &sA[aOffset]
            )[0];

            float4 a1 = reinterpret_cast<float4*>(
                &sA[aOffset + 4]
            )[0];

            regA[0] = a0.x;
            regA[1] = a0.y;
            regA[2] = a0.z;
            regA[3] = a0.w;
            regA[4] = a1.x;
            regA[5] = a1.y;
            regA[6] = a1.z;
            regA[7] = a1.w;

            float4 b0 = reinterpret_cast<float4*>(
                &sB[bOffset]
            )[0];

            float4 b1 = reinterpret_cast<float4*>(
                &sB[bOffset + 4]
            )[0];

            regB[0] = b0.x;
            regB[1] = b0.y;
            regB[2] = b0.z;
            regB[3] = b0.w;
            regB[4] = b1.x;
            regB[5] = b1.y;
            regB[6] = b1.z;
            regB[7] = b1.w;

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int globalRowC = blockRow + threadRow * TM + i;

        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalColC = blockCol + threadCol * TN + j;

            if (globalRowC < M && globalColC < N) {
                size_t cIdx =
                    static_cast<size_t>(globalRowC) *
                    static_cast<size_t>(N) +
                    static_cast<size_t>(globalColC);

                float old = beta == 0.0f ? 0.0f : C[cIdx];
                float b = bias ? bias[globalColC] : 0.0f;

                C[cIdx] = alpha * threadResults[i][j] + beta * old + b;
            }
        }
    }
}

__global__ void gemm_vectorized_bank_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    const float* __restrict__ bias,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float sA[BK * SA_STRIDE];
    __shared__ float sB[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM][TN] = {0.0f};

    float regA[TM];
    float regB[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int i = 0; i < (BM * BK) / (NUM_THREADS * 4); i++) {
            int pairIdx = threadIdx.x + i * NUM_THREADS;

            int rowA = pairIdx / (BK / 4);
            int colA4 = pairIdx % (BK / 4);

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &A[rowA * K + colA4 * 4]
                )[0];

            sA[(colA4 * 4 + 0) * SA_STRIDE + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * SA_STRIDE + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * SA_STRIDE + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * SA_STRIDE + rowA] = tmp.w;
        }

        for (int i = 0; i < (BK * BN) / (NUM_THREADS * 4); i++) {
            int pairIdx = threadIdx.x + i * NUM_THREADS;

            int rowB = pairIdx / (BN / 4);
            int colB4 = pairIdx % (BN / 4);

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &B[rowB * N + colB4 * 4]
                )[0];

            reinterpret_cast<float4*>(
                &sB[rowB * BN + colB4 * 4]
            )[0] = tmp;
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            const int aOffset =
                dotIdx * SA_STRIDE + threadRow * TM;

            const int bOffset =
                dotIdx * BN + threadCol * TN;

            const float4 a0 =
                reinterpret_cast<float4*>(&sA[aOffset])[0];

            const float4 a1 =
                reinterpret_cast<float4*>(&sA[aOffset + 4])[0];

            regA[0] = a0.x;
            regA[1] = a0.y;
            regA[2] = a0.z;
            regA[3] = a0.w;
            regA[4] = a1.x;
            regA[5] = a1.y;
            regA[6] = a1.z;
            regA[7] = a1.w;

            const float4 b0 =
                reinterpret_cast<float4*>(&sB[bOffset])[0];

            const float4 b1 =
                reinterpret_cast<float4*>(&sB[bOffset + 4])[0];

            regB[0] = b0.x;
            regB[1] = b0.y;
            regB[2] = b0.z;
            regB[3] = b0.w;
            regB[4] = b1.x;
            regB[5] = b1.y;
            regB[6] = b1.z;
            regB[7] = b1.w;

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    const int globalColBase = cCol * BN + threadCol * TN;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        float* c0_ptr =
            &C[(threadRow * TM + i) * N + threadCol * TN];

        float* c1_ptr =
            &C[(threadRow * TM + i) * N + threadCol * TN + 4];

        float b0 = bias ? bias[globalColBase + 0] : 0.0f;
        float b1 = bias ? bias[globalColBase + 1] : 0.0f;
        float b2 = bias ? bias[globalColBase + 2] : 0.0f;
        float b3 = bias ? bias[globalColBase + 3] : 0.0f;
        float b4 = bias ? bias[globalColBase + 4] : 0.0f;
        float b5 = bias ? bias[globalColBase + 5] : 0.0f;
        float b6 = bias ? bias[globalColBase + 6] : 0.0f;
        float b7 = bias ? bias[globalColBase + 7] : 0.0f;

        if (beta == 0.0f) {
            reinterpret_cast<float4*>(c0_ptr)[0] = make_float4(
                alpha * threadResults[i][0] + b0,
                alpha * threadResults[i][1] + b1,
                alpha * threadResults[i][2] + b2,
                alpha * threadResults[i][3] + b3
            );

            reinterpret_cast<float4*>(c1_ptr)[0] = make_float4(
                alpha * threadResults[i][4] + b4,
                alpha * threadResults[i][5] + b5,
                alpha * threadResults[i][6] + b6,
                alpha * threadResults[i][7] + b7
            );
        } else {
            float4 ec0 = reinterpret_cast<float4*>(c0_ptr)[0];
            float4 ec1 = reinterpret_cast<float4*>(c1_ptr)[0];

            reinterpret_cast<float4*>(c0_ptr)[0] = make_float4(
                alpha * threadResults[i][0] + beta * ec0.x + b0,
                alpha * threadResults[i][1] + beta * ec0.y + b1,
                alpha * threadResults[i][2] + beta * ec0.z + b2,
                alpha * threadResults[i][3] + beta * ec0.w + b3
            );

            reinterpret_cast<float4*>(c1_ptr)[0] = make_float4(
                alpha * threadResults[i][4] + beta * ec1.x + b4,
                alpha * threadResults[i][5] + beta * ec1.y + b5,
                alpha * threadResults[i][6] + beta * ec1.z + b6,
                alpha * threadResults[i][7] + beta * ec1.w + b7
            );
        }
    }
}

void launch_custom_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    const float* bias,
    float* C,
    cudaStream_t stream
) {
    if (can_use_decode_gemm(M, N, K)) {
        launch_decode_gemm(
            M,
            N,
            K,
            alpha,
            A,
            B,
            beta,
            bias,
            C,
            stream
        );
        return;
    }

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    dim3 block(NUM_THREADS);

    if (can_use_vectorized_bank_gemm(M, N, K, A, B, C)) {
        gemm_vectorized_bank_kernel<<<grid, block, 0, stream>>>(
            M,
            N,
            K,
            alpha,
            A,
            B,
            beta,
            bias,
            C
        );
        return;
    }

    gemm_general_kernel<<<grid, block, 0, stream>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        bias,
        C
    );
}


} // anonymous namespace

void launch_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if (use_cublas_gemm()) {
        launch_gemm_cublas_row_major(
            M,
            N,
            K,
            alpha,
            A,
            B,
            beta,
            C,
            stream
        );
        return;
    }

    launch_custom_gemm(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        nullptr,
        C,
        stream
    );
}

void launch_gemm_bias(
    int M,
    int N,
    int K,
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    if (use_cublas_gemm()) {
        launch_gemm_cublas_row_major(
            M,
            N,
            K,
            1.0f,
            A,
            B,
            0.0f,
            C,
            stream
        );

        launch_add_bias(
            C,
            bias,
            M,
            N,
            stream
        );

        return;
    }

    launch_custom_gemm(
        M,
        N,
        K,
        1.0f,
        A,
        B,
        0.0f,
        bias,
        C,
        stream
    );
}

void launch_gemm_bias_residual(
    M,
    N,
    K,
    x,
    weight,
    bias,
    y
) {
    launch_custom_gemm(
        M,
        N,
        K,
        1.0f,
        A,
        B,
        1.0f,
        bias,
        C,
        stream
    );
}


} // namespace mini_llm::kernels
