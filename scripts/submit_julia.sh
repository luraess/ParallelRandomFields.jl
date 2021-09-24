#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
export IGG_CUDAAWARE_MPI=1

export PS_THREAD_BOUND_CHECK=0
export JULIA_NUM_THREADS=4

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2
module load openmpi/gcc83-316-c112

julia_=$(which julia)

U_GPU=$1
D_VIZ=$2
D_SAVE=$3

USE_GPU=$U_GPU DO_VIZ=$D_VIZ DO_SAVE=$D_SAVE $julia_ -O3 --project --check-bounds=no runme2D_multixpu.jl
