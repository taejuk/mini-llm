#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_vectorized(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;
  const int numThreads = (BM / TM) * (BN / TN);

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  __shared__ float sA[BK * BM];
  __shared__ float sB[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  float threadResults[TM][TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  for(int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int i = 0; i < (BM * BK) / (numThreads * 4); i++) {
      int pairIdx = threadIdx.x + i * numThreads;
      int rowA = pairIdx / (BK / 4);
      int colA4 = pairIdx % (BK / 4);

      float4 tmp = reinterpret_cast<const float4*>(&A[rowA * K + colA4 * 4])[0];

      sA[(colA4*4 + 0)*BM + rowA] = tmp.x;
      sA[(colA4*4 + 1)*BM + rowA] = tmp.y;
      sA[(colA4*4 + 2)*BM + rowA] = tmp.z;
      sA[(colA4*4 + 3)*BM + rowA] = tmp.w;
    }

    for(int i = 0; i < (BK * BN) / (numThreads * 4); i++) {
      int pairIdx = threadIdx.x + i * numThreads;
      int rowB = pairIdx / (BN / 4);
      int colB4 = pairIdx % (BN / 4);

      float4 tmp = reinterpret_cast<const float4*>(&B[rowB * N + colB4 * 4])[0];
      reinterpret_cast<float4*>(&sB[rowB * BN + colB4 * 4])[0] = tmp;
    }

    __syncthreads();
    A += BK;
    B += BK * N;

    for(int dotIdx = 0; dotIdx < BK; dotIdx++) {
      float4 a0 = reinterpret_cast<float4*>(&sA[dotIdx * BM + threadRow * TM])[0];
      float4 a1 = reinterpret_cast<float4*>(&sA[dotIdx * BM + threadRow * TM+4])[0];
      regA[0] = a0.x; regA[1] = a0.y; regA[2] = a0.z; regA[3] = a0.w;
      regA[4] = a1.x; regA[5] = a1.y; regA[6] = a1.z; regA[7] = a1.w;
      
      float4 b0 = reinterpret_cast<float4*>(&sB[dotIdx * BN + threadCol * TN])[0];
      float4 b1 = reinterpret_cast<float4*>(&sB[dotIdx * BN + threadCol * TN + 4])[0];
      regB[0]=b0.x; regB[1]=b0.y; regB[2]=b0.z; regB[3]=b0.w;
      regB[4]=b1.x; regB[5]=b1.y; regB[6]=b1.z; regB[7]=b1.w;

      for(int i = 0; i < TM; i++)
        for(int j = 0; j < TN; j++) threadResults[i][j] += regA[i] * regB[j];
      
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; i++) {
    float4* c0_ptr = reinterpret_cast<float4*>
                          (&C[(threadRow * TM + i) * N + threadCol * TN]);
    float4* c1_ptr = reinterpret_cast<float4*>
                          (&C[(threadRow * TM + i) * N + threadCol * TN + 4]);
    float4 ec0 = c0_ptr[0];
    float4 ec1 = c1_ptr[0];
    c0_ptr[0] = {alpha*threadResults[i][0] + beta*ec0.x,
                  alpha*threadResults[i][1] + beta*ec0.y,
                  alpha*threadResults[i][2] + beta*ec0.z,
                  alpha*threadResults[i][3] + beta*ec0.w};
    c1_ptr[0] = {alpha*threadResults[i][4] + beta*ec1.x,
                  alpha*threadResults[i][5] + beta*ec1.y,
                  alpha*threadResults[i][6] + beta*ec1.z,
                  alpha*threadResults[i][7] + beta*ec1.w};
  }
}

void vectorized_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
  dim3 blockDim((BM / TM) * (BN / TN));  // 64
  sgemm_vectorized<<<gridDim, blockDim>>>(M, N, K, alpha ,A, B, beta,C);
}
