#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

project_dir=/scratch/lraess/Project_Julia/ParallelRandomFields/scripts/RandomField_multixpu
app_dir=./

export HOME2=${project_dir}
export JULIA_PROJECT=${HOME2}/${app_dir}
export JULIA_DEPOT_PATH=${HOME2}/julia_depot
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_MPI_BINARY=system

export IGG_CUDAAWARE_MPI=1
export JULIA_NUM_THREADS=4

module purge > /dev/null 2>&1

module load julia
module load Qt
module load cuda/11.0 
module load openmpi/gcc54-200c

julia_=$(which julia)

$julia_ -O3 --check-bounds=no RandomField3De_mulitxpu.jl

