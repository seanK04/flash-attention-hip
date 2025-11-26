#include <hip/hip_runtime.h>
#include "attention.hpp"

/*
 * Flash Attention - Memory-Efficient Attention Computation
 * =========================================================
 * 
 * MATRICES:
 *   Q (Query)  - [N, d] - What we're looking for
 *   K (Key)    - [N, d] - What we match against
 *   V (Value)  - [N, d] - What we retrieve
 *   S (Scores) - [N, N] - Attention scores: S = Q @ K^T (never fully materialized)
 *   P (Probs)  - [N, N] - Attention weights: P = softmax(S) (never fully materialized)
 *   O (Output) - [N, d] - Final result: O = P @ V
 *
 * STANDARD ATTENTION:
 *   O = softmax(Q @ K^T) @ V
 *   Problem: Q @ K^T creates an [N, N] matrix — O(N^2) memory
 *
 * FLASH ATTENTION:
 *   Process Q in tiles of size TILE_BR, K/V in tiles of size TILE_BC.
 *   For each Q tile, iterate through all K/V tiles and accumulate output
 *   using online softmax (tracking running max `m` and sum `l`).
 *   
 *   Memory: O(N) — only tile-sized chunks in shared memory
 *   Compute: Same as standard attention, just reordered
 *
 * KERNEL STRUCTURE:
 *   1. Each thread block owns one Q tile (TILE_BR rows)
 *   2. Load Q tile once into shared memory
 *   3. Loop over K/V tiles:
 *      a. Load K, V tiles into shared memory
 *      b. Compute S_tile = Q_tile @ K_tile^T
 *      c. Update running softmax stats (m, l)
 *      d. Rescale previous output accumulator
 *      e. Accumulate O += P_tile @ V_tile
 *   4. Write final output (normalized by l)
 */

// Constants
constexpr int TILE_BR = 128;
constexpr int TILE_BC = 128;
constexpr int HEAD_DIM = 128;

// Tile Loading
__device__ void load_Q_tile(const float* Q, float smem[TILE_BR][HEAD_DIM], int tile_idx, int N, int d);
__device__ void load_K_tile(const float* K, float smem[TILE_BC][HEAD_DIM], int tile_idx, int N, int d);
__device__ void load_V_tile(const float* V, float smem[TILE_BC][HEAD_DIM], int tile_idx, int N, int d);

// Matmul
__device__ void matmul_QKt(float Q[TILE_BR][HEAD_DIM], float K[TILE_BC][HEAD_DIM], float S[TILE_BR][TILE_BC]);
__device__ void matmul_PV(float P[TILE_BR][TILE_BC], float V[TILE_BC][HEAD_DIM], float O[HEAD_DIM]);

// Online Softmax
__device__ float row_max(float S[TILE_BR][TILE_BC], int row);
__device__ float row_sum_exp(float S[TILE_BR][TILE_BC], int row, float max_val);
__device__ void softmax_rescale(float O[HEAD_DIM], float m_old, float l_old, float m_new, float l_new);
__device__ void softmax_update(float* m, float* l, float m_new, float l_new);

// Output
__device__ void store_O(float* O, float reg_O[HEAD_DIM], float l, int row, int N, int d);

// Main Kernel
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    __shared__ float smem_Q[TILE_BR][HEAD_DIM];
    __shared__ float smem_K[TILE_BC][HEAD_DIM];
    __shared__ float smem_V[TILE_BC][HEAD_DIM];
    __shared__ float smem_S[TILE_BR][TILE_BC];
    
    float reg_O[HEAD_DIM] = {0};
    float m = -INFINITY;
    float l = 0;
    
    int q_tile = blockIdx.x;
    int row = q_tile * TILE_BR + threadIdx.x;
    
    load_Q_tile(Q, smem_Q, q_tile, N, d);
    __syncthreads();
    
    for (int kv_tile = 0; kv_tile < (N + TILE_BC - 1) / TILE_BC; kv_tile++) {
        load_K_tile(K, smem_K, kv_tile, N, d);
        load_V_tile(V, smem_V, kv_tile, N, d);
        __syncthreads();
        
        matmul_QKt(smem_Q, smem_K, smem_S);
        __syncthreads();
        
        float m_new = row_max(smem_S, threadIdx.x);
        float l_new = row_sum_exp(smem_S, threadIdx.x, m_new);
        
        softmax_rescale(reg_O, m, l, m_new, l_new);
        softmax_update(&m, &l, m_new, l_new);
        
        matmul_PV(smem_S, smem_V, reg_O);
        __syncthreads();
    }
    
    store_O(O, reg_O, l, row, N, d);
}

// Host
void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    dim3 grid((N + TILE_BR - 1) / TILE_BR);
    dim3 block(TILE_BR);
    hipLaunchKernelGGL(flash_attention_kernel, grid, block, 0, 0, Q, K, V, O, N, d);
}