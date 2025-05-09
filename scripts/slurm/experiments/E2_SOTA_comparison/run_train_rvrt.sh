#!/bin/bash
#SBATCH --job-name=E2_RVRT_train
#SBATCH --partition=gpu00
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=30000
#SBATCH --gres=gpu:v100:4,nvme:145
#SBATCH --nodes=1
#SBATCH --output=logs/log_train_rvrt.txt
#SBATCH --error=logs/log_train_rvrt.txt

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_CONFIG="options/experiments/E2_SOTA_comparison/archs/003_RVRT.yml"
EXP_CONFIG="options/experiments/E2_SOTA_comparison/train/003_train_RVRT.yml"
LOGGER_CONFIG="options/experiments/E2_SOTA_comparison/train/tmp_logger/003_train_RVRT.yml"


srun python basicsr/train.py --launcher slurm \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG \
    --opt $LOGGER_CONFIG