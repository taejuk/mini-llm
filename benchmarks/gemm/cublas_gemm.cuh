#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) \
  if (err != cudaSuccess) { \
      printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
      exit(1); \
  }

#define CUBLAS_CHECK(err) \
  if (err != CUBLAS_STATUS_SUCCESS) { \
      printf("cuBLAS Error: %d at line %d\n", err, __LINE__); \
      exit(1); \
  }



void cublas_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));


  CUBLAS_CHECK(cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,        // 주의: N, M 순서
      &alpha,
      B, N,           // B (N×K), ldb=N
      A, K,           // A (M×K), lda=K
      &beta,
      C, N            // C (M×N), ldc=N
  ));

  CUBLAS_CHECK(cublasDestroy(handle));
}
