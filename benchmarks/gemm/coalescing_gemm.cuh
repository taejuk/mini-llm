#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define BLOCKSIZE 32
__global__ void sgemm_coalescing(int M, int N, int K, const float *A,
                            const float *B, float *C) {
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
}

void coalescing_gemm(float* A, float* B, float* C, int M, int N, int K) {
  dim3 gridDim((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE);
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

