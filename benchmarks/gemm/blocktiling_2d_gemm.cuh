#include <cuda_runtime.h>
#include <stdio.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_blocktiling_2d(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // 총 스레드 수: (BM/TM) × (BN/TN) = 8 × 8 = 64
    const int numThreads = (BM / TM) * (BN / TN);

    // 이 스레드가 담당하는 출력 타일 내 위치
    const int threadRow = threadIdx.x / (BN / TN);  // 0..7
    const int threadCol = threadIdx.x % (BN / TN);  // 0..7

    __shared__ float sA[BM * BK];  // [64][8]
    __shared__ float sB[BK * BN];  // [8][64]

    // 이 블록의 타일 시작점으로 포인터 이동
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // 스레드당 TM×TN 누적 결과
    float threadResults[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // ── sA 로드: [BM×BK]=512 원소, 64스레드 → 스레드당 8개 ──
        for (int i = 0; i < (BM * BK) / numThreads; i++) {
            int idx  = threadIdx.x + i * numThreads;
            int rowA = idx / BK;
            int colA = idx % BK;
            sA[idx] = A[rowA * K + colA];
        }

        // ── sB 로드: [BK×BN]=512 원소, 64스레드 → 스레드당 8개 ──
        for (int i = 0; i < (BK * BN) / numThreads; i++) {
            int idx  = threadIdx.x + i * numThreads;
            int rowB = idx / BN;
            int colB = idx % BN;
            sB[idx] = B[rowB * N + colB];
        }

        __syncthreads();
        A += BK;
        B += BK * N;

        // ── 계산: dotIdx마다 outer product ──
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {

            // sA, sB → 레지스터로 캐싱
            for (int i = 0; i < TM; i++)
                regA[i] = sA[(threadRow * TM + i) * BK + dotIdx];
            for (int j = 0; j < TN; j++)
                regB[j] = sB[dotIdx * BN + threadCol * TN + j];

            // TM × TN outer product
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    threadResults[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // ── C에 결과 저장 ──
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = alpha * threadResults[i][j] + beta * C[(threadRow * TM + i) * N + threadCol * TN + j];
}

void blocktiling_2d_gemm(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blockDim((BM / TM) * (BN / TN));  // 8 × 8 = 64 스레드
    sgemm_blocktiling_2d<<<gridDim, blockDim>>>(M, N, K, alpha ,A, B, beta,C);
}
