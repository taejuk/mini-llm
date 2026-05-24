/*
 * gpu_probe.cu  ─  Dump cudaDeviceProp as JSON.
 *
 *   $ nvcc -O2 gpu_probe.cu -o gpu_probe
 *   $ ./gpu_probe            # device 0
 *   $ ./gpu_probe 1          # device 1
 *   $ ./gpu_probe > gpu_props.json
 *
 * 이 JSON 을 theoretical_roofline.py 가 읽어서 HardwareSpec 을 채운다.
 *
 * Note:
 *   - clockRate / memoryClockRate 는 CUDA 12 에서 deprecated 됐지만
 *     여전히 정상 동작함 (대안: NVML, nvmlDeviceGetClockInfo).
 *   - L2 bandwidth, Tensor Core 개수, FP32 core/SM 는 deviceProp 에
 *     노출되지 않아 compute capability lookup 으로 보강해야 함.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(_e));       \
            return EXIT_FAILURE;                                       \
        }                                                              \
    } while (0)

int main(int argc, char** argv) {
    int device = (argc > 1) ? atoi(argv[1]) : 0;

    int n_devices = 0;
    CHECK(cudaGetDeviceCount(&n_devices));
    if (device < 0 || device >= n_devices) {
        fprintf(stderr, "device %d out of range (found %d device(s))\n",
                device, n_devices);
        return EXIT_FAILURE;
    }

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, device));

    /* JSON — 의존성을 피하려고 손으로 직렬화 */
    printf("{\n");
    printf("  \"device_index\":            %d,\n", device);
    printf("  \"name\":                    \"%s\",\n", p.name);
    printf("  \"compute_capability_major\": %d,\n", p.major);
    printf("  \"compute_capability_minor\": %d,\n", p.minor);

    /* Compute */
    printf("  \"sm_count\":                %d,\n", p.multiProcessorCount);
    printf("  \"clock_rate_khz\":          %d,\n", p.clockRate);
    printf("  \"warp_size\":               %d,\n", p.warpSize);
    printf("  \"max_threads_per_sm\":      %d,\n", p.maxThreadsPerMultiProcessor);
    printf("  \"max_threads_per_block\":   %d,\n", p.maxThreadsPerBlock);
    printf("  \"regs_per_sm\":             %d,\n", p.regsPerMultiprocessor);
    printf("  \"regs_per_block\":          %d,\n", p.regsPerBlock);

    /* Memory */
    printf("  \"mem_clock_rate_khz\":      %d,\n", p.memoryClockRate);
    printf("  \"mem_bus_width_bit\":       %d,\n", p.memoryBusWidth);
    printf("  \"total_global_mem_bytes\":  %zu,\n", p.totalGlobalMem);
    printf("  \"l2_cache_size_bytes\":     %d,\n", p.l2CacheSize);
    printf("  \"shared_mem_per_block\":    %zu,\n", p.sharedMemPerBlock);
    printf("  \"shared_mem_per_sm\":       %zu,\n", p.sharedMemPerMultiprocessor);

    /* Boolean caps */
    printf("  \"ecc_enabled\":             %s,\n", p.ECCEnabled ? "true" : "false");
    printf("  \"unified_addressing\":      %s\n",  p.unifiedAddressing ? "true" : "false");
    printf("}\n");

    return EXIT_SUCCESS;
}

