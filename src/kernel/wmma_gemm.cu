#include "kernel/wmma_gemm.cuh"

__global__ void wmma_gemm_kernel(
        const __half* __restrict__ A,   /* [M, K] row-major */
        const __half* __restrict__ B,   /* [K, N] row-major */
        float*        __restrict__ C,   /* [M, N] row-major */
        int M, int N, int K)
{
    int tid = threadIdx.x;
    int bidy = blockIdx.y;
    int bidx = blockIdx.x;
    int warp_id = tid / 32;
    int warp_row = warp_id / 4;
    int warp_col = warp_id % 4;

    __shared__ __half sA[BM][BK];
    __shared__ __half sB[BK][BN];
    fragment<matrix_a, 16,16,16, __half, row_major> a_frag;
    fragment<matrix_b, 16,16,16, __half, row_major> b_frag;
    fragment<accumulator, 16,16,16, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    for(int k = 0; k < K; k += BK) {
        for(int i = tid; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int global_row = row + bidy * BM;
            int global_col = k + col;
            sA[row][col] = (global_row < M && global_col < K) ? A[global_row*K+global_col] : (__half) 0.f;  
        }
        for(int i = tid; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int global_row = k + row;
            int global_col = col + bidx * BN;
            sB[row][col] = (global_row < K && global_col < N) ? B[global_row*N + global_col] : (__half) 0.0f;
        }
        __syncthreads();
        load_matrix_sync(a_frag, &sA[warp_row*16][0], BK);
        load_matrix_sync(b_frag, &sB[0][warp_col*16], BN);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      
    }
    int c_row = bidy*BM + warp_row * 16;
    int c_col = bidx*BN + warp_col * 16;
    if(c_row < M && c_col < N) store_matrix_sync(&C[c_row * N + c_col], c_frag, N, mem_row_major);

}

void wmma_gemm(const __half* A, const __half* B, float* C,
               int M, int N, int K)
{
    dim3 grid((N + BN - 1) / BN,
          (M + BM - 1) / BM);
    dim3 block(THREADS_PER_BLOCK);
    wmma_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}