#!/bin/bash
#SBATCH --job-name=E1_test
#SBATCH --partition=gpusamll
#SBATCH --time=0-08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --output=logs/log_test_%a.txt
#SBATCH --error=logs/log_test_%a.txt
#SBATCH --array=0-35

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_CONFIG="options/experiments/E1_depth_impact/archs/shiftnet.yml"
DEFAULTS_CONFIG="options/experiments/E1_depth_impact/test/defaults.yml"

EXP_ROOT="options/experiments/E1_depth_impact/test"
declare -a EXP_CONFIG_FOLDERS=("configs00" "configs01" "configs02")
EXP_CONFIG_LIST=()
for folder in "${EXP_CONFIG_FOLDERS[@]}"
do
    CONFIGS=($(ls $EXP_ROOT/$folder))
    for i in "${!CONFIGS[@]}"
    do
        CONFIGS[$i]="$folder/${CONFIGS[$i]}"
    done
    EXP_CONFIG_LIST+=("${CONFIGS[@]}")
done
EXP_CONFIG=${EXP_CONFIG_LIST[$SLURM_ARRAY_TASK_ID]}


srun python basicsr/test.py --launcher none \
    --opt $DEFAULTS_CONFIG \
    --opt $ARCH_CONFIG \
    --opt $EXP_ROOT/$EXP_CONFIG