#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2
module load openmpi/gcc83-316-c112

mpirun_=$(which mpirun)

USE_GPU=true
DO_VIZ=true
DO_SAVE=false

$mpirun_ -np 4 -rf gpu_rankfile_node40 ./submit_julia.sh $USE_GPU $DO_VIZ $DO_SAVE