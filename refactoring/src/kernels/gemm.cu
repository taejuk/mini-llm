#include "kernels/gemm.cuh"

#include <stdint.h>

namespace mini_llm::kernels {

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

__global__ void gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int numThreads = (BM / TM) * (BN / TN);

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];

    const int blockRow = cRow * BM;
    const int blockCol = cCol * BN;

    float threadResults[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ------------------------------------------------------------
        // Load A tile
        // A: [M, K]
        // sA layout: [BK, BM]
        // ------------------------------------------------------------
        const int numAFloat4 = (BM * BK) / 4;

        for (int i = threadIdx.x; i < numAFloat4; i += numThreads) {
            int linear4 = i;

            int rowA = linear4 / (BK / 4);
            int colA4 = linear4 % (BK / 4);

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

            sA[(colA4 * 4 + 0) * BM + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * BM + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * BM + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * BM + rowA] = tmp.w;
        }

        // ------------------------------------------------------------
        // Load B tile
        // B: [K, N]
        // sB layout: [BK, BN]
        // ------------------------------------------------------------
        const int numBFloat4 = (BK * BN) / 4;

        for (int i = threadIdx.x; i < numBFloat4; i += numThreads) {
            int linear4 = i;

            int rowB = linear4 / (BN / 4);
            int colB4 = linear4 % (BN / 4);

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

        // ------------------------------------------------------------
        // Compute
        // ------------------------------------------------------------
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float4 a0 = reinterpret_cast<float4*>(
                &sA[dotIdx * BM + threadRow * TM]
            )[0];

            float4 a1 = reinterpret_cast<float4*>(
                &sA[dotIdx * BM + threadRow * TM + 4]
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
                &sB[dotIdx * BN + threadCol * TN]
            )[0];

            float4 b1 = reinterpret_cast<float4*>(
                &sB[dotIdx * BN + threadCol * TN + 4]
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

    // ------------------------------------------------------------
    // Store C
    // C = alpha * A * B + beta * C
    // ------------------------------------------------------------
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

                C[cIdx] =
                    alpha * threadResults[i][j] +
                    beta * C[cIdx];
            }
        }
    }
}

__global__ void gemm_bias_kernel(
    int M,
    int N,
    int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int numThreads = (BM / TM) * (BN / TN);

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float sA[BK * BM];
    __shared__ float sB[BK * BN];

    const int blockRow = cRow * BM;
    const int blockCol = cCol * BN;

    float threadResults[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ------------------------------------------------------------
        // Load A tile
        // ------------------------------------------------------------
        const int numAFloat4 = (BM * BK) / 4;

        for (int i = threadIdx.x; i < numAFloat4; i += numThreads) {
            int linear4 = i;

            int rowA = linear4 / (BK / 4);
            int colA4 = linear4 % (BK / 4);

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

            sA[(colA4 * 4 + 0) * BM + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * BM + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * BM + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * BM + rowA] = tmp.w;
        }

        // ------------------------------------------------------------
        // Load B tile
        // ------------------------------------------------------------
        const int numBFloat4 = (BK * BN) / 4;

        for (int i = threadIdx.x; i < numBFloat4; i += numThreads) {
            int linear4 = i;

            int rowB = linear4 / (BN / 4);
            int colB4 = linear4 % (BN / 4);

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

        // ------------------------------------------------------------
        // Compute
        // ------------------------------------------------------------
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float4 a0 = reinterpret_cast<float4*>(
                &sA[dotIdx * BM + threadRow * TM]
            )[0];

            float4 a1 = reinterpret_cast<float4*>(
                &sA[dotIdx * BM + threadRow * TM + 4]
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
                &sB[dotIdx * BN + threadCol * TN]
            )[0];

            float4 b1 = reinterpret_cast<float4*>(
                &sB[dotIdx * BN + threadCol * TN + 4]
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

    // ------------------------------------------------------------
    // Store C
    // C = A * B + bias
    // ------------------------------------------------------------
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

                float b = bias ? bias[globalColC] : 0.0f;

                C[cIdx] = threadResults[i][j] + b;
            }
        }
    }
}

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

    constexpr int numThreads = (BM / TM) * (BN / TN);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    dim3 block(numThreads);

    gemm_kernel<<<grid, block, 0, stream>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        C
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

    constexpr int numThreads = (BM / TM) * (BN / TN);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    dim3 block(numThreads);

    gemm_bias_kernel<<<grid, block, 0, stream>>>(
        M,
        N,
        K,
        A,
        B,
        bias,
        C
    );
}

} // namespace mini_llm::kernels