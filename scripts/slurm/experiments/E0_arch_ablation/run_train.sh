#!/bin/bash
#SBATCH --job-name=E0_train
#SBATCH --partition=gpu
#SBATCH --time=1-15:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:v100:4,nvme:225
#SBATCH --nodes=1
#SBATCH --output=logs/log_train_%a.txt
#SBATCH --error=logs/log_train_%a.txt
#SBATCH --array=0-32

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

DEFAULTS="options/experiments/E0_arch_ablation/train/defaults.yml"

EXP_ROOT="options/experiments/E0_arch_ablation/train/configs"
EXP_CONFIG_LIST=( $(ls $EXP_ROOT) )
EXP_CONFIG=${EXP_CONFIG_LIST[$SLURM_ARRAY_TASK_ID]}

srun python basicsr/train.py \
    --launcher slurm \
    --opt $DEFAULTS \
    --opt $EXP_ROOT/$EXP_CONFIG