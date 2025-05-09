#!/bin/bash
#SBATCH --job-name=E2_EDVR_train
#SBATCH --partition=gpu
#SBATCH --time=2-23:59:59
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=30000
#SBATCH --gres=gpu:v100:4,nvme:145
#SBATCH --nodes=1
#SBATCH --output=logs/log_train_edvr.txt
#SBATCH --error=logs/log_train_edvr.txt

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_CONFIG="options/experiments/E2_SOTA_comparison/archs/001_EDVR.yml"
EXP_CONFIG="options/experiments/E2_SOTA_comparison/train/001_train_EDVR.yml"


srun python basicsr/train.py --launcher slurm \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG 