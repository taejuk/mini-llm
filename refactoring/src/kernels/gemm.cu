#include "kernels/gemm.cuh"

namespace mini_llm::kernels {

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
        // Load A tile: A[M, K] -> sA[BK, BM]
        //
        // 기존 float4 global load는 K가 4의 배수가 아닐 때
        // row start가 16-byte align되지 않아 misaligned address 가능.
        // 따라서 global load는 scalar로 안전하게 처리.
        // ------------------------------------------------------------
        const int numAFloat4 = (BM * BK) / 4;

        for (int i = threadIdx.x; i < numAFloat4; i += numThreads) {
            int linear4 = i;

            int rowA = linear4 / (BK / 4);
            int colA4 = linear4 % (BK / 4);

            int globalRowA = blockRow + rowA;
            int globalColA = bkIdx + colA4 * 4;

            float4 tmp;
            tmp.x = 0.0f;
            tmp.y = 0.0f;
            tmp.z = 0.0f;
            tmp.w = 0.0f;

            if (globalRowA < M) {
                size_t base =
                    static_cast<size_t>(globalRowA) *
                    static_cast<size_t>(K) +
                    static_cast<size_t>(globalColA);

                if (globalColA + 0 < K) {
                    tmp.x = A[base + 0];
                }
                if (globalColA + 1 < K) {
                    tmp.y = A[base + 1];
                }
                if (globalColA + 2 < K) {
                    tmp.z = A[base + 2];
                }
                if (globalColA + 3 < K) {
                    tmp.w = A[base + 3];
                }
            }

            sA[(colA4 * 4 + 0) * BM + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * BM + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * BM + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * BM + rowA] = tmp.w;
        }

        // ------------------------------------------------------------
        // Load B tile: B[K, N] -> sB[BK, BN]
        //
        // N이 4의 배수가 아닐 수 있으므로 B도 scalar load.
        // GPT2 vocab size 50257도 4의 배수가 아니어서 중요함.
        // ------------------------------------------------------------
        const int numBFloat4 = (BK * BN) / 4;

        for (int i = threadIdx.x; i < numBFloat4; i += numThreads) {
            int linear4 = i;

            int rowB = linear4 / (BN / 4);
            int colB4 = linear4 % (BN / 4);

            int globalRowB = bkIdx + rowB;
            int globalColB = blockCol + colB4 * 4;

            float4 tmp;
            tmp.x = 0.0f;
            tmp.y = 0.0f;
            tmp.z = 0.0f;
            tmp.w = 0.0f;

            if (globalRowB < K) {
                size_t base =
                    static_cast<size_t>(globalRowB) *
                    static_cast<size_t>(N) +
                    static_cast<size_t>(globalColB);

                if (globalColB + 0 < N) {
                    tmp.x = B[base + 0];
                }
                if (globalColB + 1 < N) {
                    tmp.y = B[base + 1];
                }
                if (globalColB + 2 < N) {
                    tmp.z = B[base + 2];
                }
                if (globalColB + 3 < N) {
                    tmp.w = B[base + 3];
                }
            }

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
        // Load A tile safely
        // ------------------------------------------------------------
        const int numAFloat4 = (BM * BK) / 4;

        for (int i = threadIdx.x; i < numAFloat4; i += numThreads) {
            int linear4 = i;

            int rowA = linear4 / (BK / 4);
            int colA4 = linear4 % (BK / 4);

            int globalRowA = blockRow + rowA;
            int globalColA = bkIdx + colA4 * 4;

            float4 tmp;
            tmp.x = 0.0f;
            tmp.y = 0.0f;
            tmp.z = 0.0f;
            tmp.w = 0.0f;

            if (globalRowA < M) {
                size_t base =
                    static_cast<size_t>(globalRowA) *
                    static_cast<size_t>(K) +
                    static_cast<size_t>(globalColA);

                if (globalColA + 0 < K) {
                    tmp.x = A[base + 0];
                }
                if (globalColA + 1 < K) {
                    tmp.y = A[base + 1];
                }
                if (globalColA + 2 < K) {
                    tmp.z = A[base + 2];
                }
                if (globalColA + 3 < K) {
                    tmp.w = A[base + 3];
                }
            }

            sA[(colA4 * 4 + 0) * BM + rowA] = tmp.x;
            sA[(colA4 * 4 + 1) * BM + rowA] = tmp.y;
            sA[(colA4 * 4 + 2) * BM + rowA] = tmp.z;
            sA[(colA4 * 4 + 3) * BM + rowA] = tmp.w;
        }

        // ------------------------------------------------------------
        // Load B tile safely
        // ------------------------------------------------------------
        const int numBFloat4 = (BK * BN) / 4;

        for (int i = threadIdx.x; i < numBFloat4; i += numThreads) {
            int linear4 = i;

            int rowB = linear4 / (BN / 4);
            int colB4 = linear4 % (BN / 4);

            int globalRowB = bkIdx + rowB;
            int globalColB = blockCol + colB4 * 4;

            float4 tmp;
            tmp.x = 0.0f;
            tmp.y = 0.0f;
            tmp.z = 0.0f;
            tmp.w = 0.0f;

            if (globalRowB < K) {
                size_t base =
                    static_cast<size_t>(globalRowB) *
                    static_cast<size_t>(N) +
                    static_cast<size_t>(globalColB);

                if (globalColB + 0 < N) {
                    tmp.x = B[base + 0];
                }
                if (globalColB + 1 < N) {
                    tmp.y = B[base + 1];
                }
                if (globalColB + 2 < N) {
                    tmp.z = B[base + 2];
                }
                if (globalColB + 3 < N) {
                    tmp.w = B[base + 3];
                }
            }

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
    // Store C = A x B + bias
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
