/*
 * flash_attention.cu  —  warp-reduction 버전
 *
 * 이전 버전 vs 이번 버전:
 *
 *   block 레이아웃:
 *     이전: (Br=32,)      — 1 thread 가 Q 1행 전담, 내적 64 ops 직렬
 *     이번: (32, Br=32)   — 1 warp(32 lane) 가 Q 1행 전담
 *
 *   Q·K^T 내적:
 *     이전: thread 1개가 for k=0..63 순차 곱셈
 *     이번: 32 lane 이 k 차원을 stride-32 로 분담 (각 2 elem, d=64)
 *           → __shfl_down_sync() 5단계로 합산 (log2(32)=5)
 *
 *   P·V 누적:
 *     이전: thread 1개가 Bc×d 이중루프 직렬
 *     이번: 32 lane 이 출력 차원 d 를 stride-32 로 분담
 *
 * 공유 메모리 (48KB 한계 내):
 *   sQ [Br * d]   = 32*64*4 =  8KB
 *   sK [Bc * d]   = 32*64*4 =  8KB
 *   sV [Bc * d]   =           8KB
 *   sO [Br * d]   =           8KB
 *   sS [Br * Bc]  = 32*32*4 =  4KB
 *   합계                       36KB  ✓
 *
 * 컴파일:
 *   nvcc -O2 -arch=sm_86 src/kernels/flash_attention.cu -o flash_attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAT_N 512
#define MAT_D 64
#define Br    32
#define Bc    32

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* ============================================================
 * warp_reduce_sum
 * ============================================================
 * 32개 lane 의 val 을 5단계 butterfly shuffle 로 합산.
 * 반환값: lane 0 에 전체 합, 나머지 lane 은 부분합(읽지 않음).
 *
 * __shfl_down_sync(mask, val, delta):
 *   lane i 가 lane (i+delta) 의 val 을 받아온다.
 *   mask=0xffffffff → 32개 lane 전부 참여.
 * ============================================================ */
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);
    return val;   /* lane 0 이 최종 합을 가짐 */
}

/* ============================================================
 * flash_attention_kernel
 *
 * Grid : dim3(Tr)         Tr = ceil(n / Br)
 * Block: dim3(32, Br)     32 = WARP_SIZE, Br = Q tile 행 수
 *
 *   threadIdx.x = lane     (0..31) : warp 내 위치
 *   threadIdx.y = warp_row (0..Br-1): 이 warp 가 담당하는 Q tile 행
 * ============================================================ */
__global__ void flash_attention_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float* O, int n, int d)
{
    int lane     = threadIdx.x;           /* warp lane  0..31       */
    int warp_row = threadIdx.y;           /* Q tile 내 행  0..Br-1  */
    int bid      = blockIdx.x;            /* Q tile 번호            */
    int i_abs    = bid * Br + warp_row;   /* 전체 Q 행 절대 인덱스  */

    /* ── 공유 메모리 파티션 ──────────────────────────────────────── */
    extern __shared__ float smem[];
    float* sQ = smem;                     /* [Br * d]  */
    float* sK = sQ + Br * d;             /* [Bc * d]  */
    float* sV = sK + Bc * d;             /* [Bc * d]  */
    float* sO = sV + Bc * d;             /* [Br * d]  */
    float* sS = sO + Br * d;             /* [Br * Bc] — dot 결과/P 재활용 */

    /* flat thread index: 전체 로드에서 사용 */
    int flat     = warp_row * 32 + lane;
    int nthreads = blockDim.x * blockDim.y;  /* 32 * Br = 1024 */

    /* ── sQ 로드 (1024 threads 협력) ────────────────────────────── */
    for (int idx = flat; idx < Br * d; idx += nthreads) {
        int r = idx / d, c = idx % d;
        int g = bid * Br + r;
        sQ[idx] = (g < n) ? Q[g * d + c] : 0.f;
    }

    /* ── sO 초기화: 각 lane 이 자기 담당 열만 초기화 ─────────────── */
    for (int k = lane; k < d; k += 32)
        sO[warp_row * d + k] = 0.f;

    float l = 0.f;            /* 누적 softmax 분모  (per warp) */
    float m = -INFINITY;      /* 누적 max           (per warp) */
    const float scale = 1.f / sqrtf((float)d);

    __syncthreads();

    /* ── Tc 개 K tile 순회 ──────────────────────────────────────── */
    int Tc = (n + Bc - 1) / Bc;

    for (int j = 0; j < Tc; j++) {

        /* ── sK, sV 로드 ─────────────────────────────────────────── */
        for (int idx = flat; idx < Bc * d; idx += nthreads) {
            int r = idx / d, c = idx % d;
            int g = j * Bc + r;
            sK[idx] = (g < n) ? K[g * d + c] : 0.f;
            sV[idx] = (g < n) ? V[g * d + c] : 0.f;
        }
        __syncthreads();

        /* ── Q·K^T  (warp reduction) ─────────────────────────────
         *
         * 내적 sQ[warp_row] · sK[c] 를 32 lane 이 협력하여 계산.
         *
         *  step 1) 각 lane 이 k = lane, lane+32 (d=64 이므로 2 elem) 담당
         *          partial 에 부분합 누적
         *  step 2) warp_reduce_sum() → lane 0 에 전체 합
         *  step 3) lane 0 이 sS 에 기록
         *
         * 이후 __syncwarp() 로 lane 0 의 sS 쓰기를 warp 전체에 가시화.
         * ─────────────────────────────────────────────────────── */
        for (int c = 0; c < Bc; c++) {
            float partial = 0.f;
            for (int k = lane; k < d; k += 32)
                partial += sQ[warp_row * d + k] * sK[c * d + k];

            partial = warp_reduce_sum(partial);   /* ← __shfl_down_sync ×5 */

            if (lane == 0)
                sS[warp_row * Bc + c] = partial * scale;
        }
        __syncwarp();   /* lane 0 의 sS 쓰기를 같은 warp 모든 lane 에 노출 */

        /* ── online softmax 갱신 ─────────────────────────────────
         * 모든 lane 이 동일한 sS 값을 읽어 m, l 을 독립 계산.
         * warp 내 m, l 은 값이 일치하므로 broadcast 불필요.
         * ─────────────────────────────────────────────────────── */
        float m_prev = m;
        float m_next = m;
        for (int c = 0; c < Bc; c++)
            m_next = fmaxf(m_next, sS[warp_row * Bc + c]);

        /* 이전 누적값 재스케일 */
        float rescale = expf(m_prev - m_next);
        l *= rescale;
        for (int k = lane; k < d; k += 32)
            sO[warp_row * d + k] *= rescale;

        /* exp → P, l 에 합산 */
        for (int c = 0; c < Bc; c++) {
            float p = expf(sS[warp_row * Bc + c] - m_next);
            sS[warp_row * Bc + c] = p;   /* sS 를 P 로 재활용 */
            l += p;
        }
        m = m_next;

        /* ── P·V 누적 ────────────────────────────────────────────
         * 각 lane 이 출력 열 k = lane, lane+32 (stride 32) 담당.
         * ─────────────────────────────────────────────────────── */
        for (int k = lane; k < d; k += 32) {
            float pv = 0.f;
            for (int c = 0; c < Bc; c++)
                pv += sS[warp_row * Bc + c] * sV[c * d + k];
            sO[warp_row * d + k] += pv;
        }
        __syncthreads();   /* 다음 tile 의 sK/sV 로드 전에 sO 쓰기 완료 보장 */
    }

    /* ── 출력 정규화 후 global memory 저장 ─────────────────────── */
    if (i_abs < n) {
        float inv_l = 1.f / l;
        for (int k = lane; k < d; k += 32)
            O[i_abs * d + k] = sO[warp_row * d + k] * inv_l;
    }
}

/* ============================================================
 * 호출 래퍼
 * ============================================================ */
void flash_attention(float* Q, float* K, float* V, float* O, int n, int d)
{
    int Tr = (n + Br - 1) / Br;
    dim3 grid(Tr);
    dim3 block(32, Br);   /* (WARP_SIZE, Br) */

    /* 공유 메모리: sQ + sK + sV + sO + sS */
    size_t smem = ((size_t)(Br + Bc + Bc + Br) * d
                   + (size_t)Br * Bc) * sizeof(float);

    flash_attention_kernel<<<grid, block, smem>>>(Q, K, V, O, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ============================================================
 * main — 정확도 검증 + 타이밍
 * ============================================================ */
int main(void)
{
    printf("=== FlashAttention2 (warp reduction) ===\n");
    printf("N=%d  d=%d  Br=%d  Bc=%d\n\n", MAT_N, MAT_D, Br, Bc);

    int n = MAT_N, d = MAT_D;
    size_t sz = (size_t)n * d * sizeof(float);

    float *h_Q = (float*)malloc(sz);
    float *h_K = (float*)malloc(sz);
    float *h_V = (float*)malloc(sz);
    float *h_O = (float*)malloc(sz);

    srand(42);
    for (int i = 0; i < n * d; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, sz));
    CUDA_CHECK(cudaMalloc(&d_K, sz));
    CUDA_CHECK(cudaMalloc(&d_V, sz));
    CUDA_CHECK(cudaMalloc(&d_O, sz));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz, cudaMemcpyHostToDevice));

    /* warmup */
    flash_attention(d_Q, d_K, d_V, d_O, n, d);

    /* 타이밍 */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    flash_attention(d_Q, d_K, d_V, d_O, n, d);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("execution time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(h_O, d_O, sz, cudaMemcpyDeviceToHost));

    /* 이전 버전의 O_flash.bin 과 비교하려면:
     *   python3 -c "import numpy as np; \
     *     a=np.fromfile('O_flash.bin',np.float32); \
     *     b=np.fromfile('O_flash_new.bin',np.float32); \
     *     print('max diff:', np.max(np.abs(a-b)))"
     */
    FILE* f;
    f = fopen("Q_flash.bin","wb"); fwrite(h_Q,sizeof(float),n*d,f); fclose(f);
    f = fopen("K_flash.bin","wb"); fwrite(h_K,sizeof(float),n*d,f); fclose(f);
    f = fopen("V_flash.bin","wb"); fwrite(h_V,sizeof(float),n*d,f); fclose(f);
    f = fopen("O_flash.bin","wb"); fwrite(h_O,sizeof(float),n*d,f); fclose(f);
    printf("Saved Q,K,V,O → bin files\n");

    free(h_Q); free(h_K); free(h_V); free(h_O);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

