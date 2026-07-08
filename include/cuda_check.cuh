#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(err__), __FILE__, __LINE__);        \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

#ifdef MINI_LLM_DEBUG
  #define CUDA_CHECK_KERNEL()                                              \
      do { CUDA_CHECK(cudaGetLastError());                                 \
           CUDA_CHECK(cudaDeviceSynchronize()); } while (0)
#else
  #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif