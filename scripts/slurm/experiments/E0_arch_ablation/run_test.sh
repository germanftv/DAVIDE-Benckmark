#!/bin/bash
#SBATCH --job-name=E0_test
#SBATCH --partition=gpusmall
#SBATCH --time=0-03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --output=logs/log_test_%a.txt
#SBATCH --error=logs/log_test_%a.txt
#SBATCH --array=0-32

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

DEFAULTS="options/experiments/E0_arch_ablation/test/defaults.yml"

EXP_ROOT="options/experiments/E0_arch_ablation/test/configs"
EXP_CONFIG_LIST=( $(ls $EXP_ROOT) )
EXP_CONFIG=${EXP_CONFIG_LIST[$SLURM_ARRAY_TASK_ID]}

srun python basicsr/test.py \
    --launcher none \
    --opt $DEFAULTS \
    --opt $EXP_ROOT/$EXP_CONFIG