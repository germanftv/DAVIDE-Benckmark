#!/bin/bash
#SBATCH --job-name=E2_EDVR_test
#SBATCH --partition=gpusmall
#SBATCH --time=0-04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --output=logs/log_test_edvr.txt
#SBATCH --error=logs/log_test_edvr.txt

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_CONFIG="options/experiments/E2_SOTA_comparison/archs/001_EDVR.yml"
EXP_CONFIG="options/experiments/E2_SOTA_comparison/test/001_test_EDVR.yml"


srun python basicsr/test.py \
    --launcher none \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG 