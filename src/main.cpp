#include <mpi.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include "attention.hpp"

/*
 * Flash Attention MPI Driver
 * ==========================
 * 
 * PARALLELIZATION STRATEGY:
 *   - Distribute attention heads across MPI ranks
 *   - Each rank computes attention for its assigned heads on its GPU
 *   - Rank 0 gathers results at the end
 * 
 * COMMUNICATION PATTERN:
 *   1. Rank 0 initializes Q, K, V matrices
 *   2. Scatter: Rank 0 sends each rank its head subset
 *   3. Compute: Each rank independently runs Flash Attention on GPU
 *   4. Gather: All ranks send results back to rank 0
 */

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void check_mpi(int err, const char* msg) {
    if (err != MPI_SUCCESS) {
        std::cerr << "MPI Error: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device (assuming 4 GPUs per node)
    int num_devices;
    hipGetDeviceCount(&num_devices);
    check_hip(hipSetDevice(rank % num_devices), "hipSetDevice");

    // Problem parameters
    const int N = 16384;           // Sequence length
    const int d = 64;             // Head dimension
    const int num_heads = 64;     // Total number of attention heads
    const int batch_size = 1;     // Batch size
    
    // Distribute heads across MPI ranks
    if (num_heads % size != 0 && rank == 0) {
        std::cerr << "Warning: num_heads (" << num_heads 
                  << ") not evenly divisible by size (" << size << ")" << std::endl;
    }
    
    const int heads_per_rank = (num_heads + size - 1) / size;
    const int my_head_start = rank * heads_per_rank;
    const int my_head_end = std::min(my_head_start + heads_per_rank, num_heads);
    const int my_num_heads = my_head_end - my_head_start;
    
    if (rank == 0) {
    std::cout << "=== Flash Attention MPI Configuration ===" << std::endl;
    std::cout << "MPI ranks: " << size << std::endl;
    std::cout << "Sequence length: " << N << std::endl;
    std::cout << "Head dimension: " << d << std::endl;
    std::cout << "Total heads: " << num_heads << std::endl;
    std::cout << "Heads per rank: " << heads_per_rank << std::endl;
    std::cout << "# of GPUs: " << num_devices << std::endl;
    }
    
    // Rank 0 initializes and broadcasts input data
    float *h_Q = nullptr, *h_K = nullptr, *h_V = nullptr;
    float *h_O_full = nullptr;
    
    if (rank == 0) {
        // Allocate full Q, K, V on rank 0 [batch_size, num_heads, N, d]
        size_t size_qkv = batch_size * num_heads * N * d * sizeof(float);
        h_Q = new float[batch_size * num_heads * N * d];
        h_K = new float[batch_size * num_heads * N * d];
        h_V = new float[batch_size * num_heads * N * d];
        h_O_full = new float[batch_size * num_heads * N * d];
        
        // Initialize with random values
        for (size_t i = 0; i < batch_size * num_heads * N * d; i++) {
            h_Q[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            h_K[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            h_V[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        
        std::cout << "Input data initialized" << std::endl;
    }
    
    // Allocate local buffers for this rank's heads
    size_t local_size = batch_size * my_num_heads * N * d;
    float *h_Q_local = new float[local_size];
    float *h_K_local = new float[local_size];
    float *h_V_local = new float[local_size];
    float *h_O_local = new float[local_size];
    
    // Scatter Q, K, V from rank 0 to all ranks
    if (rank == 0) {
        // Copy rank 0's portion
        size_t offset = 0;
        std::copy(h_Q + offset * N * d, h_Q + (offset + my_num_heads) * N * d, h_Q_local);
        std::copy(h_K + offset * N * d, h_K + (offset + my_num_heads) * N * d, h_K_local);
        std::copy(h_V + offset * N * d, h_V + (offset + my_num_heads) * N * d, h_V_local);
        
        // Send to other ranks
        for (int r = 1; r < size; r++) {
            int r_head_start = r * heads_per_rank;
            int r_head_end = std::min(r_head_start + heads_per_rank, num_heads);
            int r_num_heads = r_head_end - r_head_start;
            
            if (r_num_heads > 0) {
                size_t r_offset = r_head_start * N * d;
                size_t r_size = r_num_heads * N * d;
                
                MPI_Send(h_Q + r_offset, r_size, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
                MPI_Send(h_K + r_offset, r_size, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
                MPI_Send(h_V + r_offset, r_size, MPI_FLOAT, r, 2, MPI_COMM_WORLD);
            }
        }
    } else {
        // Receive data from rank 0
        if (my_num_heads > 0) {
            MPI_Recv(h_Q_local, local_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(h_K_local, local_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(h_V_local, local_size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate GPU memory
    float *d_Q, *d_K, *d_V, *d_O;
    check_hip(hipMalloc(&d_Q, local_size * sizeof(float)), "hipMalloc Q");
    check_hip(hipMalloc(&d_K, local_size * sizeof(float)), "hipMalloc K");
    check_hip(hipMalloc(&d_V, local_size * sizeof(float)), "hipMalloc V");
    check_hip(hipMalloc(&d_O, local_size * sizeof(float)), "hipMalloc O");
    
    // Copy to GPU
    check_hip(hipMemcpy(d_Q, h_Q_local, local_size * sizeof(float), hipMemcpyHostToDevice), "copy Q to device");
    check_hip(hipMemcpy(d_K, h_K_local, local_size * sizeof(float), hipMemcpyHostToDevice), "copy K to device");
    check_hip(hipMemcpy(d_V, h_V_local, local_size * sizeof(float), hipMemcpyHostToDevice), "copy V to device");
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Launch kernel for each head this rank owns
    for (int h = 0; h < my_num_heads; h++) {
        float *Q_head = d_Q + h * N * d;
        float *K_head = d_K + h * N * d;
        float *V_head = d_V + h * N * d;
        float *O_head = d_O + h * N * d;
        
        flash_attention_forward(Q_head, K_head, V_head, O_head, N, d);
    }
    
    // Wait for GPU completion
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << "Computation time: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
    }
    
    // Copy results back to host
    check_hip(hipMemcpy(h_O_local, d_O, local_size * sizeof(float), hipMemcpyDeviceToHost), "copy O to host");
    
    // Gather results on rank 0
    if (rank == 0) {
        // Copy rank 0's portion
        std::copy(h_O_local, h_O_local + local_size, h_O_full);
        
        // Receive from other ranks
        for (int r = 1; r < size; r++) {
            int r_head_start = r * heads_per_rank;
            int r_head_end = std::min(r_head_start + heads_per_rank, num_heads);
            int r_num_heads = r_head_end - r_head_start;
            
            if (r_num_heads > 0) {
                size_t r_offset = r_head_start * N * d;
                size_t r_size = r_num_heads * N * d;
                
                MPI_Recv(h_O_full + r_offset, r_size, MPI_FLOAT, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        std::cout << "Results gathered on rank 0" << std::endl;
    } else {
        // Send results to rank 0
        if (my_num_heads > 0) {
            MPI_Send(h_O_local, local_size, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
        }
    }
    
    // Cleanup
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);
    
    delete[] h_Q_local;
    delete[] h_K_local;
    delete[] h_V_local;
    delete[] h_O_local;
    
    if (rank == 0) {
        delete[] h_Q;
        delete[] h_K;
        delete[] h_V;
        delete[] h_O_full;
    }
    
    MPI_Finalize();
    return 0;
}