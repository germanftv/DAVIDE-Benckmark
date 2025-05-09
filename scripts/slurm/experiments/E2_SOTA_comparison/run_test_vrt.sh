#!/bin/bash
#SBATCH --job-name=E2_VRT_test
#SBATCH --partition=gpusmall
#SBATCH --time=0-04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --output=logs/log_test_vrt.txt
#SBATCH --error=logs/log_test_vrt.txt

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_CONFIG="options/experiments/E2_SOTA_comparison/archs/002_VRT.yml"
EXP_CONFIG="options/experiments/E2_SOTA_comparison/test/002_test_VRT.yml"


srun python basicsr/test.py \
    --launcher none \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG 