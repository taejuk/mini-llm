#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define BLOCKSIZE 32
__global__ void sgemm_naive(int M, int N, int K, const float *A,
                            const float *B, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
}

void naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
  dim3 gridDim((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE,1);
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
