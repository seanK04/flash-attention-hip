/*
Naive (Regular) Attention Implementation with MPI and HIP.

This implementation explicitly materializes the N x N attention score matrix.
It uses MPI to distribute multiple attention heads across available GPUs/Processes.

Algorithm:
1. S = Q * K^T     (Size: N x N)
2. P = Softmax(S)  (Size: N x N)
3. O = P * V       (Size: N x d)
*/

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

// Configuration
constexpr int N = 1024;       // Sequence Length
constexpr int d = 256;         // Head Dimension
constexpr int TOTAL_HEADS = 16; // Total heads to process across all ranks

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Kernel 1: Compute Scores S = Q * K^T
// Q: [N, d], K: [N, d] -> S: [N, N]
__global__ void kernel_score(const float* __restrict__ Q, const float* __restrict__ K, float* __restrict__ S, int n, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..N-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    if (row < n && col < n) {
        float val = 0.0f;
        for (int k = 0; k < dim; k++) {
            val += Q[row * dim + k] * K[col * dim + k];
        }
        S[row * n + col] = val;
    }
}

// Kernel 2: Compute Softmax P = softmax(S) row-wise
// S: [N, N] -> P: [N, N]
__global__ void kernel_softmax(const float* __restrict__ S, float* __restrict__ P, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        // 1. Find max for numerical stability
        float max_val = -1e30f;
        for (int j = 0; j < n; j++) {
            max_val = fmaxf(max_val, S[row * n + j]);
        }

        // 2. Exponentials and sum
        float sum = 0.0f;
        // We can compute P in place or in separate buffer. Using separate P for clarity.
        for (int j = 0; j < n; j++) {
            float val = expf(S[row * n + j] - max_val);
            P[row * n + j] = val;
            sum += val;
        }

        // 3. Normalize
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < n; j++) {
            P[row * n + j] *= inv_sum;
        }
    }
}

// Kernel 3: Compute Output O = P * V
// P: [N, N], V: [N, d] -> O: [N, d]
__global__ void kernel_value(const float* __restrict__ P, const float* __restrict__ V, float* __restrict__ O, int n, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..N-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..d-1

    if (row < n && col < dim) {
        float val = 0.0f;
        for (int k = 0; k < n; k++) {
            val += P[row * n + k] * V[k * dim + col];
        }
        O[row * dim + col] = val;
    }
}

// CPU Reference for verification
void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& O_ref) {
    std::vector<float> S(N * N);
    std::vector<float> P(N * N);

    // Q * K^T
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) sum += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = sum;
        }
    }

    // Softmax
    for (int i = 0; i < N; ++i) {
        float max_v = -1e30f;
        for (int j = 0; j < N; ++j) max_v = std::max(max_v, S[i * N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            P[i * N + j] = std::exp(S[i * N + j] - max_v);
            sum += P[i * N + j];
        }
        for (int j = 0; j < N; ++j) P[i * N + j] /= sum;
    }

    // P * V
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) sum += P[i * N + k] * V[k * d + j];
            O_ref[i * d + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign GPU
    int num_devices;
    hipGetDeviceCount(&num_devices);
    HIP_CHECK(hipSetDevice(rank % num_devices));

    if (rank == 0) {
        std::cout << "Naive Attention MPI Benchmark\n";
        std::cout << "N=" << N << ", d=" << d << ", Total Heads=" << TOTAL_HEADS 
                  << ", MPI Size=" << size << std::endl;
    }

    // Calculate workload
    if (TOTAL_HEADS % size != 0) {
        if(rank==0) std::cerr << "Error: Total Heads must be divisible by MPI Size." << std::endl;
        MPI_Finalize();
        return 1;
    }
    int local_heads = TOTAL_HEADS / size;

    size_t size_matrix_small = N * d * sizeof(float);
    size_t size_matrix_large = N * N * sizeof(float);

    // Host Memory
    // We allocate enough for all local heads to store input/output
    std::vector<float> h_Q(local_heads * N * d);
    std::vector<float> h_K(local_heads * N * d);
    std::vector<float> h_V(local_heads * N * d);
    std::vector<float> h_O(local_heads * N * d);

    // Initialize Data (Random)
    std::mt19937 gen(42 + rank); // Seed depends on rank
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (auto& val : h_Q) val = dis(gen);
    for (auto& val : h_K) val = dis(gen);
    for (auto& val : h_V) val = dis(gen);

    // Device Memory
    // We only need ONE set of buffers on the GPU, we will loop through heads
    // This simulates processing heads sequentially on one GPU to save memory.
    // Naive attention uses O(N^2) memory, so we reuse buffers to be safe.
    float *d_Q, *d_K, *d_V, *d_O; // Small [N, d]
    float *d_S, *d_P;             // Large [N, N]

    HIP_CHECK(hipMalloc(&d_Q, size_matrix_small));
    HIP_CHECK(hipMalloc(&d_K, size_matrix_small));
    HIP_CHECK(hipMalloc(&d_V, size_matrix_small));
    HIP_CHECK(hipMalloc(&d_O, size_matrix_small));
    HIP_CHECK(hipMalloc(&d_S, size_matrix_large));
    HIP_CHECK(hipMalloc(&d_P, size_matrix_large));

    // Synchronize before timing
    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Loop over local heads
    for (int h = 0; h < local_heads; ++h) {
        // 1. Copy Input for this head
        HIP_CHECK(hipMemcpy(d_Q, &h_Q[h * N * d], size_matrix_small, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_K, &h_K[h * N * d], size_matrix_small, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_V, &h_V[h * N * d], size_matrix_small, hipMemcpyHostToDevice));

        // 2. Kernel 1: S = Q * K^T
        dim3 block(16, 16);
        dim3 grid_score((N + 15) / 16, (N + 15) / 16);
        kernel_score<<<grid_score, block>>>(d_Q, d_K, d_S, N, d);

        // 3. Kernel 2: P = Softmax(S)
        // One thread per row for simplicity in naive implementation
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        kernel_softmax<<<blocks, threads>>>(d_S, d_P, N);

        // 4. Kernel 3: O = P * V
        dim3 grid_out((d + 15) / 16, (N + 15) / 16);
        kernel_value<<<grid_out, block>>>(d_P, d_V, d_O, N, d);

        // 5. Copy Output back
        HIP_CHECK(hipMemcpy(&h_O[h * N * d], d_O, size_matrix_small, hipMemcpyDeviceToHost));
    }

    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Total Execution Time: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
    }

    // 6. Gather all results to Rank 0 (To make it fully testable/comparable)
    std::vector<float> h_O_global;
    if (rank == 0) {
        h_O_global.resize(TOTAL_HEADS * N * d);
    }
    
    // Each rank sends its local_heads block of data
    MPI_Gather(
        h_O.data(),             // Send buffer
        local_heads * N * d,    // Send count
        MPI_FLOAT,              // Send type
        h_O_global.data(),      // Recv buffer
        local_heads * N * d,    // Recv count (per process)
        MPI_FLOAT,              // Recv type
        0,                      // Root
        MPI_COMM_WORLD
    );

    // Verify Correctness (Rank 0 verifies Head 0)
    if (rank == 0) {
        std::cout << "Verifying Head 0 on CPU..." << std::endl;
        std::vector<float> h_O_ref(N * d);
        
        // Extract Head 0 data (from the global gathered buffer or local if rank 0 processed head 0)
        // Since the global buffer is laid out sequentially by rank, and rank 0 processes heads [0..local_heads-1],
        // we can take the first chunk.
        
        // However, for verification, we need the original inputs for Head 0.
        // Rank 0 definitely has inputs for Head 0 in its local h_Q/K/V.
        
        std::vector<float> head_Q(h_Q.begin(), h_Q.begin() + N*d);
        std::vector<float> head_K(h_K.begin(), h_K.begin() + N*d);
        std::vector<float> head_V(h_V.begin(), h_V.begin() + N*d);

        cpu_attention(head_Q, head_K, head_V, h_O_ref);

        // Compare against the gathered result (which should match local h_O for rank 0)
        float max_diff = 0.0f;
        for (int i = 0; i < N * d; ++i) {
            float diff = std::abs(h_O_global[i] - h_O_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        std::cout << "Max Error: " << max_diff << (max_diff < 1e-4 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Cleanup
    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    hipFree(d_S); hipFree(d_P);
    
    MPI_Finalize();
    return 0;
}