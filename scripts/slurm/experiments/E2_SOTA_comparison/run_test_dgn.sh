#!/bin/bash
#SBATCH --job-name=E2_DGN_test
#SBATCH --partition=gpusmall
#SBATCH --time=0-04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --output=logs/log_test_dgn_%a.txt
#SBATCH --error=logs/log_test_dgn_%a.txt
#SBATCH --array=1,3

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_MULTIFRAME_CONFIG="options/experiments/E2_SOTA_comparison/archs/000_DGN_multi_frame.yml"
ARCH_SINGLEFRAME_CONFIG="options/experiments/E2_SOTA_comparison/archs/000_DGN_single_frame.yml"

EXP_PATTERN='000_test_DGN*'
DIR_PATH="options/experiments/E2_SOTA_comparison/test"
EXP_CONFIG_LIST=( $(ls $DIR_PATH/$EXP_PATTERN) )
EXP_CONFIG=${EXP_CONFIG_LIST[$SLURM_ARRAY_TASK_ID]}

# check if there is "single_frame" in the EXP_CONFIG
if grep -q "single_frame" $EXP_CONFIG; then
    ARCH_CONFIG=$ARCH_SINGLEFRAME_CONFIG
else
    ARCH_CONFIG=$ARCH_MULTIFRAME_CONFIG
fi

srun python basicsr/test.py \
    --launcher none \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG 