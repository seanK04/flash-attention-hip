#include <mpi.h>
#include <hip/hip_runtime.h>
#include "attention.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    hipSetDevice(rank % 4);
    
    // TODO: allocate memory, call kernel, gather results
    
    MPI_Finalize();
    return 0;
}