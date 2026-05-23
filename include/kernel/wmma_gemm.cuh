#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define BM 64    
#define BN 64    
#define BK 16    

#define WARPS_M (BM / 16)
#define WARPS_N (BN / 16)
#define WARPS_PER_BLOCK (WARPS_M * WARPS_N)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

__global__ void wmma_gemm_kernel(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float*        __restrict__ C,
        int M, int N, int K
);

void wmma_gemm(const __half* A, const __half* B, float* C, int M, int N, int K);
