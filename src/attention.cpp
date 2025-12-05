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
    // Each thread loads THREAD_TILE_M rows
    for (int i = 0; i < THREAD_TILE_M; i++) {
        int row = tile_idx * TILE_BC + threadIdx.y * THREAD_TILE_M + i;
        
        for (int j = threadIdx.x; j < d; j += THREADS_X) {
            if (row < N && j < d) {
                smem[threadIdx.y * THREAD_TILE_M + i][j] = K[row * d + j];
            } else {
                smem[threadIdx.y * THREAD_TILE_M + i][j] = 0.0f;
            }
        }
    }
}

__device__ void load_V_tile(const float* V, float smem[TILE_BC][HEAD_DIM], int tile_idx, int N, int d) {
    // Same structure as load_K_tile (V has same dimensions as K)
    for (int i = 0; i < THREAD_TILE_M; i++) {
        int row = tile_idx * TILE_BC + threadIdx.y * THREAD_TILE_M + i;
        
        for (int j = threadIdx.x; j < d; j += THREADS_X) {
            if (row < N && j < d) {
                smem[threadIdx.y * THREAD_TILE_M + i][j] = V[row * d + j];
            } else {
                smem[threadIdx.y * THREAD_TILE_M + i][j] = 0.0f;
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

__device__ void matmul_PV(float P[TILE_BR][TILE_BC], float V[TILE_BC][HEAD_DIM], float O[THREAD_TILE_M][THREAD_TILE_N]) {
    int row_start = threadIdx.y * THREAD_TILE_M;
    int col_start = threadIdx.x * THREAD_TILE_N;
    #pragma unroll
    for (int k = 0; k < TILE_BC; k++) {
        float p_vals[THREAD_TILE_M];
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            p_vals[i] = P[row_start + i][k];
        }
        float v_vals[THREAD_TILE_N];
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            v_vals[j] = V[k][col_start + j];
        }
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_M; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                O[i][j] += p_vals[i] * v_vals[j];
            }
        }
    }
}

// Online Softmax 
__device__ float warp_reduce_max(float val) {
    // Reduce across the 16 threads in the x-dimension
    // These threads are contiguous in the warp (lanes differ by 1)
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, offset);
    }
    return val;
}

// Each thread computes max over its 4 columns for each of its 4 rows
// Returns array of 4 LOCAL maxes (not yet reduced across threads)
__device__ void local_row_max(float S[TILE_BR][TILE_BC], float m_local[THREAD_TILE_M]) {
    int row_start = threadIdx.y * THREAD_TILE_M;
    int col_start = threadIdx.x * THREAD_TILE_N;
    
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        float max_val = -INFINITY;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            max_val = fmaxf(max_val, S[row_start + i][col_start + j]);
        }
        m_local[i] = max_val;
    }
}

// After getting local max, reduce across threads to get true row max,
// then compute exp(S - max) and local sum, then reduce sum across threads.
// Also overwrites S with exp values for use in matmul_PV.
__device__ void compute_softmax_stats(
    float S[TILE_BR][TILE_BC],
    float m_new[THREAD_TILE_M],    // Output: new max for each row
    float l_new[THREAD_TILE_M]     // Output: new sum for each row
) {
    int row_start = threadIdx.y * THREAD_TILE_M;
    int col_start = threadIdx.x * THREAD_TILE_N;
    
    // Step 1: Compute local max for this thread's columns
    float m_local[THREAD_TILE_M];
    local_row_max(S, m_local);
    
    // Step 2: Reduce to get global row max
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        m_new[i] = warp_reduce_max(m_local[i]);
    }
    
    // Step 3: Compute exp(S - max) for this thread's columns and local sum
    float l_local[THREAD_TILE_M] = {0};
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            float val = expf(S[row_start + i][col_start + j] - m_new[i]);
            S[row_start + i][col_start + j] = val;  // Store for matmul_PV
            l_local[i] += val;
        }
    }
    
    // Step 4: Reduce to get global row sum
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        l_new[i] = warp_reduce_sum(l_local[i]);
    }
}

__device__ void softmax_rescale(
    float O[THREAD_TILE_M][THREAD_TILE_N],
    float l[THREAD_TILE_M],
    float m_old[THREAD_TILE_M],
    float m_new[THREAD_TILE_M]
) {
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        float scale = expf(m_old[i] - m_new[i]);
        // Rescale both the output accumulator and the running sum
        l[i] *= scale;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            O[i][j] *= scale;
        }
    }
}

__device__ void softmax_update(
    float m[THREAD_TILE_M],
    float l[THREAD_TILE_M],
    float m_new[THREAD_TILE_M],
    float l_new[THREAD_TILE_M]
) {
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        l[i] += l_new[i];  // l was already rescaled in softmax_rescale
        m[i] = m_new[i];
    }
}

// Output
__device__ void store_O(float* O, float reg_O[THREAD_TILE_M][THREAD_TILE_N], float l[THREAD_TILE_M], int q_tile, int N, int d) {
    int row_start = q_tile * TILE_BR + threadIdx.y * THREAD_TILE_M;
    int col_start = threadIdx.x * THREAD_TILE_N;

    for (int i = 0; i < THREAD_TILE_M; i++) {
        int row = row_start + i; 
        if (row < N) {
            for (int j = 0; j < THREAD_TILE_N; j++) {
                O[row * d + col_start + j] = reg_O[i][j] / l[i];
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
    float reg_O[THREAD_TILE_M][THREAD_TILE_N] = {0};
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
        
        matmul_QKt(smem_Q, smem_K, smem_S);
        __syncthreads();
        
        // Compute softmax stats and overwrite smem_S with exp values
        float m_new[THREAD_TILE_M];
        float l_new[THREAD_TILE_M];
        compute_softmax_stats(smem_S, m_new, l_new);
        // No sync needed here, each thread only reads/writes its own rows
        
        // Rescale previous output (must happen before updating m)
        softmax_rescale(reg_O, l, m, m_new);
        
        // Update running stats
        softmax_update(m, l, m_new, l_new);
        
        __syncthreads();  // Ensure smem_S writes from compute_softmax_stats are visible
        
        matmul_PV(smem_S, smem_V, reg_O);
        __syncthreads();  // Before next iteration loads new K/V
    }

    store_O(O, reg_O, l, q_tile, N, d);
}

// Host
void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    // Grid dimension: one block per Q tile
    dim3 grid((N + TILE_BR - 1) / TILE_BR);
    
    // Block dimension: 16 x 16 = 256 threads
    dim3 block(THREADS_X, THREADS_Y);
    
    // Launch kernel
    hipLaunchKernelGGL(flash_attention_kernel, grid, block, 0, 0, Q, K, V, O, N, d);
}
