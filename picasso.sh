#!/usr/bin/env bash
# The name to show in queue lists for this job:
#SBATCH -J Q_Learning

# Number of desired cpus:
#SBATCH --cpus-per-task=10

# Amount of RAM needed for this job:
#SBATCH --mem=32gb

# The time the job will be running:
#SBATCH --time=10:00:00

# To use GPUs you have to request them:
##SBATCH --gres=gpu:1
#SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=./output/error.%J.err
#SBATCH --output=./output/output.%J.out

# Load your virtual environment as a module
source load /mnt/home/users/tic_102_2_uma/fedeloz/Q-Learning/SatNEx/bin/activate

# the program to execute with its parameters:
#time ./mi_programa argumentos${SLURM_ARRAYID}.jpg > out_$SLURM_ARRAYID.out
hostname
# hostname prints the name of the node that the job is running on
# time Measures the time taken to execute the program.

# $SLURM_CPUS_PER_TASK has the same value requested in --cpus-per-task 
time python ./SimulationRL.py -t $SLURM_CPUS_PER_TASK 

