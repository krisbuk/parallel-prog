#!/bin/bash
#SBATCH --job-name=my_matrix_job
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --output=my_out_%j.log

module purge
module load intel/mpi4

N=$1
A_FILE="A_${N}.txt"
B_FILE="B_${N}.txt"

mpirun -np $SLURM_NTASKS ./mult $N $A_FILE $B_FILE