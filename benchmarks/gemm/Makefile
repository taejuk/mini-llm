NVCC = nvcc
ARCH     = -arch=sm_70
FLAGS    = -O3 $(ARCH) -I. -lcublas -lnvToolsExt -lineinfo
TARGET   = gemm_bench

$(TARGET): main.cu native_gemm.cuh coalescing_gemm.cuh caching_gemm.cuh \
           blocktiling_1d_gemm.cuh blocktiling_2d_gemm.cuh \
           vectorized_gemm.cuh vectorized_bank_gemm.cuh \
           cublas_gemm.cuh wmma_gemm.cuh shared_tensor_core.cuh \
           tensor_core_v2_gemm.cuh
	$(NVCC) $(FLAGS) -o $(TARGET) main.cu

clean:
	rm -f $(TARGET)