#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#define BLOCKSIZE 32
__global__ void sgemm_caching(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
  __shared__ float sA[BLOCKSIZE*BLOCKSIZE];
  __shared__ float sB[BLOCKSIZE*BLOCKSIZE];
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int aOffset = cRow * K * BLOCKSIZE;
  const int bOffset = cCol * BLOCKSIZE;

  const int threadx = threadIdx.x;
  const int thready = threadIdx.y;

  float tmp = 0.0;
  
  for(int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    //sA[threadx * BLOCKSIZE + thready] = A[aOffset + bkIdx + thready + K * threadx];
    sA[thready * BLOCKSIZE + threadx] = A[aOffset + K*threadx + bkIdx + thready];
    sB[threadx * BLOCKSIZE + thready] = B[bOffset + N * (bkIdx + threadx) + thready];
    __syncthreads();

    for(int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) 
	    //tmp += sA[threadx * BLOCKSIZE + dotIdx] * sB[dotIdx * BLOCKSIZE + thready];
            tmp += sA[dotIdx * BLOCKSIZE + threadx] * sB[dotIdx * BLOCKSIZE + thready];
    __syncthreads();
  }
  C[(cRow * BLOCKSIZE + threadx)*N + cCol * BLOCKSIZE + thready] = alpha * tmp + beta * C[(cRow * BLOCKSIZE + threadx)*N + cCol * BLOCKSIZE + thready];

}

void caching_gemm(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  dim3 gridDim((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE);
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

  sgemm_caching<<<gridDim, blockDim>>>(M, N, K, alpha ,A, B, beta,C);

}
