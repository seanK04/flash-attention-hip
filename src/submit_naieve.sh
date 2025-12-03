#!/bin/bash
#SBATCH --job-name=attention_benchmark
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=mi2508x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

set -e

echo "Running on host: $(hostname)"

# --- MODULE LOADING ---
# 1. Load ROCm
module load rocm

# 2. Load OpenMPI (Using --ignore_cache as suggested by your error log)
echo "Loading OpenMPI..."
# We try to load it. If it fails, we print a warning but try to proceed 
# (sometimes it's already loaded by the environment)
module --ignore_cache load "openmpi" || echo "Warning: 'module load openmpi' returned an error. Checking if mpic++ exists anyway..."

# 3. List loaded modules for debugging
module list

# --- COMPILER CHECK ---
if ! command -v mpic++ &> /dev/null; then
    echo "Error: mpic++ could not be found."
    echo "Please run 'module spider openmpi' in your terminal to find the correct module name"
    echo "and update the 'module load' line in this script."
    exit 1
fi

# OMP Settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Verify GPUs are actually visible
echo "Checking visible GPUs..."
rocm-smi

echo "Compiling (MPI + HIP)..."

# Compilation
# We use $(mpic++ --showme...) to safely find MPI includes/libs for hipcc
hipcc -O3 naieve_attention_implementation.cpp \
      -o naieve_attn \
      -I$(mpic++ --showme:incdirs | cut -d' ' -f1) \
      -L$(mpic++ --showme:libdirs | cut -d' ' -f1) \
      -lmpi

echo "Running with ${SLURM_NTASKS} MPI ranks..."

# Run the benchmark
srun ./naieve_attn > results_benchmark.txt

echo "Done! Check results_benchmark.txt for output."