#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define BLOCKSIZE 32
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x*N + y];
  }
}

void naive_gemm(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  dim3 gridDim((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE,1);
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha ,A, B, beta,C);
}
