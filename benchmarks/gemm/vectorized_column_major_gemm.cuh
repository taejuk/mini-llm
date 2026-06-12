#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define VCM_BM 64
#define VCM_BN 64
#define VCM_BK 8
#define VCM_TM 8
#define VCM_TN 8

#define VCM_SA_STRIDE (VCM_BM + 4)
#define VCM_NUM_THREADS ((VCM_BM / VCM_TM) * (VCM_BN / VCM_TN))

#define VCM_TRANSPOSE_TILE_DIM 32
#define VCM_TRANSPOSE_BLOCK_ROWS 8

__global__ void transpose_A_tiled_kernel(
    int M,
    int K,
    const float* __restrict__ A,
    float* __restrict__ AT
) {
    __shared__ float tile[VCM_TRANSPOSE_TILE_DIM][VCM_TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * VCM_TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * VCM_TRANSPOSE_TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < VCM_TRANSPOSE_TILE_DIM; j += VCM_TRANSPOSE_BLOCK_ROWS) {
        if (x < K && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] =
                A[static_cast<size_t>(y + j) * K + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * VCM_TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * VCM_TRANSPOSE_TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < VCM_TRANSPOSE_TILE_DIM; j += VCM_TRANSPOSE_BLOCK_ROWS) {
        if (x < M && (y + j) < K) {
            AT[static_cast<size_t>(y + j) * M + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void transpose_A_for_vectorized_column_major_a(
    int M,
    int K,
    const float* A,
    float* AT
) {
    dim3 block(
        VCM_TRANSPOSE_TILE_DIM,
        VCM_TRANSPOSE_BLOCK_ROWS
    );

    dim3 grid(
        (K + VCM_TRANSPOSE_TILE_DIM - 1) / VCM_TRANSPOSE_TILE_DIM,
        (M + VCM_TRANSPOSE_TILE_DIM - 1) / VCM_TRANSPOSE_TILE_DIM
    );

    transpose_A_tiled_kernel<<<grid, block>>>(M, K, A, AT);
}

__global__ void vectorized_column_major_gemm_kernel(
    int M,
    int N,
    int K,
    float alpha,
    const float* __restrict__ AT,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (VCM_BN / VCM_TN);
    const int threadCol = threadIdx.x % (VCM_BN / VCM_TN);

    __shared__ float sA[VCM_BK * VCM_SA_STRIDE];
    __shared__ float sB[VCM_BK * VCM_BN];

    C += static_cast<size_t>(cRow) * VCM_BM * N + cCol * VCM_BN;

    float threadResults[VCM_TM][VCM_TN] = {0.0f};

    float regA[VCM_TM];
    float regB[VCM_TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += VCM_BK) {
        constexpr int A_VEC_PER_K = VCM_BM / 4;
        constexpr int A_VEC_TOTAL = VCM_BK * A_VEC_PER_K;

        for (int i = 0; i < A_VEC_TOTAL / VCM_NUM_THREADS; i++) {
            int vecIdx = threadIdx.x + i * VCM_NUM_THREADS;

            int localK = vecIdx / A_VEC_PER_K;
            int rowA4 = vecIdx % A_VEC_PER_K;

            int globalK = bkIdx + localK;
            int globalRow = cRow * VCM_BM + rowA4 * 4;

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &AT[static_cast<size_t>(globalK) * M + globalRow]
                )[0];

            reinterpret_cast<float4*>(
                &sA[localK * VCM_SA_STRIDE + rowA4 * 4]
            )[0] = tmp;
        }

        constexpr int B_VEC_PER_ROW = VCM_BN / 4;
        constexpr int B_VEC_TOTAL = VCM_BK * B_VEC_PER_ROW;

        for (int i = 0; i < B_VEC_TOTAL / VCM_NUM_THREADS; i++) {
            int vecIdx = threadIdx.x + i * VCM_NUM_THREADS;

            int rowB = vecIdx / B_VEC_PER_ROW;
            int colB4 = vecIdx % B_VEC_PER_ROW;

            int globalK = bkIdx + rowB;
            int globalCol = cCol * VCM_BN + colB4 * 4;

            const float4 tmp =
                reinterpret_cast<const float4*>(
                    &B[static_cast<size_t>(globalK) * N + globalCol]
                )[0];

            reinterpret_cast<float4*>(
                &sB[rowB * VCM_BN + colB4 * 4]
            )[0] = tmp;
        }

        __syncthreads();

        #pragma unroll
        for (int dotIdx = 0; dotIdx < VCM_BK; dotIdx++) {
            const int aOffset =
                dotIdx * VCM_SA_STRIDE + threadRow * VCM_TM;

            const int bOffset =
                dotIdx * VCM_BN + threadCol * VCM_TN;

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
            for (int i = 0; i < VCM_TM; i++) {
                #pragma unroll
                for (int j = 0; j < VCM_TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < VCM_TM; i++) {
        float* c0_ptr =
            &C[(threadRow * VCM_TM + i) * N + threadCol * VCM_TN];

        float* c1_ptr =
            &C[(threadRow * VCM_TM + i) * N + threadCol * VCM_TN + 4];

        if (beta == 0.0f) {
            reinterpret_cast<float4*>(c0_ptr)[0] = make_float4(
                alpha * threadResults[i][0],
                alpha * threadResults[i][1],
                alpha * threadResults[i][2],
                alpha * threadResults[i][3]
            );

            reinterpret_cast<float4*>(c1_ptr)[0] = make_float4(
                alpha * threadResults[i][4],
                alpha * threadResults[i][5],
                alpha * threadResults[i][6],
                alpha * threadResults[i][7]
            );
        } else {
            float4 ec0 = reinterpret_cast<float4*>(c0_ptr)[0];
            float4 ec1 = reinterpret_cast<float4*>(c1_ptr)[0];

            reinterpret_cast<float4*>(c0_ptr)[0] = make_float4(
                alpha * threadResults[i][0] + beta * ec0.x,
                alpha * threadResults[i][1] + beta * ec0.y,
                alpha * threadResults[i][2] + beta * ec0.z,
                alpha * threadResults[i][3] + beta * ec0.w
            );

            reinterpret_cast<float4*>(c1_ptr)[0] = make_float4(
                alpha * threadResults[i][4] + beta * ec1.x,
                alpha * threadResults[i][5] + beta * ec1.y,
                alpha * threadResults[i][6] + beta * ec1.z,
                alpha * threadResults[i][7] + beta * ec1.w
            );
        }
    }
}

void vectorized_column_major_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* AT,
    const float* B,
    float beta,
    float* C
) {
    if ((M % VCM_BM) != 0 ||
        (N % VCM_BN) != 0 ||
        (K % VCM_BK) != 0) {
        printf(
            "vectorized_column_major_gemm requires M %% %d == 0, "
            "N %% %d == 0, K %% %d == 0. Got M=%d, N=%d, K=%d\n",
            VCM_BM,
            VCM_BN,
            VCM_BK,
            M,
            N,
            K
        );
        return;
    }

    dim3 gridDim(
        (N + VCM_BN - 1) / VCM_BN,
        (M + VCM_BM - 1) / VCM_BM
    );

    dim3 blockDim(VCM_NUM_THREADS);

    vectorized_column_major_gemm_kernel<<<gridDim, blockDim>>>(
        M,
        N,
        K,
        alpha,
        AT,
        B,
        beta,
        C
    );
}
