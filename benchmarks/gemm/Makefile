NVCC = nvcc
ARCH     = -arch=sm_70
FLAGS    = -O3 $(ARCH) -I. -lcublas
TARGET   = gemm_bench

$(TARGET): main.cu native_gemm.cuh coalescing_gemm.cuh caching_gemm.cuh
	$(NVCC) $(FLAGS) -o $(TARGET) main.cu

clean:
	rm -f $(TARGET)
