#!/bin/bash
#SBATCH --job-name=E2_VRT_train
#SBATCH --partition=gpumedium
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:4,nvme:225
#SBATCH --nodes=1
#SBATCH --output=logs/log_train_vrt.txt
#SBATCH --error=logs/log_train_vrt.txt

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_CONFIG="options/experiments/E2_SOTA_comparison/archs/002_VRT.yml"
EXP_CONFIG="options/experiments/E2_SOTA_comparison/train/002_train_VRT.yml"
LOGGER_CONFIG="options/experiments/E2_SOTA_comparison/train/tmp_logger/002_train_VRT.yml"


srun python basicsr/train.py --launcher slurm \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG \
    --opt $LOGGER_CONFIG