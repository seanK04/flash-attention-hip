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
constexpr int TILE_BR = 64;
constexpr int TILE_BC = 64;
constexpr int HEAD_DIM = 64;

// Thread block configuration
constexpr int THREADS_X = 16;  // columns
constexpr int THREADS_Y = 16;  // rows
constexpr int THREAD_TILE_M = TILE_BR / THREADS_Y;  // 4 rows per thread
constexpr int THREAD_TILE_N = TILE_BC / THREADS_X;  // 4 cols per thread

// Tile Loading
__device__ void load_Q_tile(const float* Q, float smem[TILE_BR][HEAD_DIM], int tile_idx, int N, int d) {
    // Each thread loads THREAD_TILE_M rows
    // threadIdx.y determines which rows this thread is responsible for
    for (int i = 0; i < THREAD_TILE_M; i++) {
        int row = tile_idx * TILE_BR + threadIdx.y * THREAD_TILE_M + i;
        
        // Each thread loads multiple columns using threadIdx.x with striding
        for (int j = threadIdx.x; j < d; j += THREADS_X) {
            if (row < N && j < d) {
                smem[threadIdx.y * THREAD_TILE_M + i][j] = Q[row * d + j];
            } else {
                smem[threadIdx.y * THREAD_TILE_M + i][j] = 0.0f;
            }
        }
    }
}

__device__ void load_K_tile(const float* K, float smem[TILE_BC][HEAD_DIM], int tile_idx, int N, int d) {
    // Each thread loads THREAD_TILE_N rows
    for (int i = 0; i < THREAD_TILE_N; i++) {
        int row = tile_idx * TILE_BC + threadIdx.x * THREAD_TILE_N + i;
        
        for (int j = threadIdx.y; j < d; j += THREADS_Y) {
            if (row < N && j < d) {
                smem[threadIdx.x * THREAD_TILE_N + i][j] = K[row * d + j];
            } else {
                smem[threadIdx.x * THREAD_TILE_N + i][j] = 0.0f;
            }
        }
    }
}

__device__ void load_V_tile(const float* V, float smem[TILE_BC][HEAD_DIM], int tile_idx, int N, int d) {
    // Same structure as load_K_tile (V has same dimensions as K)
    for (int i = 0; i < THREAD_TILE_N; i++) {
        int row = tile_idx * TILE_BC + threadIdx.x * THREAD_TILE_N + i;
        
        for (int j = threadIdx.y; j < d; j += THREADS_Y) {
            if (row < N && j < d) {
                smem[threadIdx.x * THREAD_TILE_N + i][j] = V[row * d + j];
            } else {
                smem[threadIdx.x * THREAD_TILE_N + i][j] = 0.0f;
            }
        }
    }
}

// Matmul
__device__ void matmul_QKt(float Q[TILE_BR][HEAD_DIM], float K[TILE_BC][HEAD_DIM], float S[TILE_BR][TILE_BC]) {
    // Thread (tx, ty) handles:
    //  Rows: [ty * THREAD_TILE_M, ty * THREAD_TILE_M + THREAD_TILE_M]
    //  Cols: [tx * THREAD_TILE_N, tx * THREAD_TILE_N + THREAD_TILE_N]
    
    int row_start = threadIdx.y * THREAD_TILE_M;
    int col_start = threadIdx.x * THREAD_TILE_N;

    float acc[THREAD_TILE_M][THREAD_TILE_N] = {0};

    // Loop over the reduction dimension: S[i,j] = sum_k Q[i,k] * K[j,k]
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; k++) {
        // Load Q values for this thread's rows at column k
        float q_vals[THREAD_TILE_M];
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            q_vals[i] = Q[row_start + i][k];
        }
        // Load K values for this thread's columns at column k
        float k_vals[THREAD_TILE_N];
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            k_vals[j] = K[col_start + j][k];
        }
        // Compute outer product contribution to the 4×4 micro-tile
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                acc[i][j] += q_vals[i] * k_vals[j];
            }
        }
    }
    // Write the micro-tile results back to shared memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            S[row_start + i][col_start + j] = acc[i][j];
        }
    }
}

__device__ void matmul_PV(float P[TILE_BR][TILE_BC], float V[TILE_BC][HEAD_DIM], float O[THREAD_TILE_M][HEAD_DIM]) {
    int row_start = threadIdx.y * THREAD_TILE_M;

    // Each thread handles THREAD_TILE_M rows x HEAD_DIM columns
    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float sum = 0.0f;
            // Reduction over TILE_BC
            for (int j = 0; j < TILE_BC; j++) {
                sum += P[row_start + i][j] * V[j][d];
            }
            O[i][d] += sum; // Accumulate
        }
    }
}

// Online Softmax (TODO: implement with per-thread-tile operations)
__device__ float row_max(float S[TILE_BR][TILE_BC], int row);
__device__ float row_sum_exp(float S[TILE_BR][TILE_BC], int row, float max_val);
__device__ void softmax_rescale(float O[THREAD_TILE_M][HEAD_DIM], float m_old[THREAD_TILE_M], float l_old[THREAD_TILE_M], float m_new[THREAD_TILE_M], float l_new[THREAD_TILE_M]);
__device__ void softmax_update(float m[THREAD_TILE_M], float l[THREAD_TILE_M], float m_new[THREAD_TILE_M], float l_new[THREAD_TILE_M]);

// Output
__device__ void store_O(float* O, float reg_O[THREAD_TILE_M][HEAD_DIM], float l[THREAD_TILE_M], int q_tile, int N, int d) {
    int row_start = q_tile * TILE_BR + threadIdx.y *THREAD_TILE_M;

    for (int i = 0; i < THREAD_TILE_M; i++) {
        int row = row_start + i; 
        if (row < N) {
            for (int j = 0; j < HEAD_DIM; j++) {
                // Normalize by softmax sum and write
                O[row * d + j] = reg_O[i][j] / l[i];
            }
        }
    }
}

// Main Kernel
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    __shared__ float smem_Q[TILE_BR][HEAD_DIM];
    __shared__ float smem_K[TILE_BC][HEAD_DIM];
    __shared__ float smem_V[TILE_BC][HEAD_DIM];
    __shared__ float smem_S[TILE_BR][TILE_BC];
    
    // Each thread accumulates THREAD_TILE_M rows of output
    float reg_O[THREAD_TILE_M][HEAD_DIM] = {0};
    float m[THREAD_TILE_M];
    float l[THREAD_TILE_M];
    
    // Initialize running stats for each row this thread handles
    for (int i = 0; i < THREAD_TILE_M; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    
    int q_tile = blockIdx.x;
    // This thread handles rows: q_tile * TILE_BR + threadIdx.y * THREAD_TILE_M + [0..THREAD_TILE_M-1]
    
    load_Q_tile(Q, smem_Q, q_tile, N, d);
    __syncthreads();
    
    for (int kv_tile = 0; kv_tile < (N + TILE_BC - 1) / TILE_BC; kv_tile++) {
        load_K_tile(K, smem_K, kv_tile, N, d);
        load_V_tile(V, smem_V, kv_tile, N, d);
        __syncthreads();
        
        // TODO: matmul_QKt(smem_Q, smem_K, smem_S);
        // __syncthreads();
        
        // TODO: Compute m_new and l_new for each of THREAD_TILE_M rows
        // TODO: softmax_rescale(reg_O, m, l, m_new, l_new);
        // TODO: softmax_update(m, l, m_new, l_new);
        
        // TODO: matmul_PV(smem_S, smem_V, reg_O);
        // __syncthreads();
    }
    
    // TODO: store_O(O, reg_O, l, q_tile, N, d);
}

// Host
void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    dim3 grid((N + TILE_BR - 1) / TILE_BR);
    dim3 block(THREADS_X, THREADS_Y);  // 16x16 = 256 threads per block
    hipLaunchKernelGGL(flash_attention_kernel, grid, block, 0, 0, Q, K, V, O, N, d);
}