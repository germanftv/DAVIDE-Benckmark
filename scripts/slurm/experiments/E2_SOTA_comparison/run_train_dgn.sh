#!/bin/bash
#SBATCH --job-name=E2_DGN_train
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=30000
#SBATCH --gres=gpu:v100:2,nvme:225
#SBATCH --nodes=1
#SBATCH --output=logs/log_train_dgn_%a.txt
#SBATCH --error=logs/log_train_dgn_%a.txt
#SBATCH --array=0-1

# Load CUDA
module load cuda
# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

cd ../../../..

ARCH_MULTIFRAME_CONFIG="options/experiments/E2_SOTA_comparison/archs/000_DGN_multi_frame.yml"
ARCH_SINGLEFRAME_CONFIG="options/experiments/E2_SOTA_comparison/archs/000_DGN_single_frame.yml"

EXP_PATTERN='000_train_DGN*'
DIR_PATH="options/experiments/E2_SOTA_comparison/train"
EXP_CONFIG_LIST=( $(ls $DIR_PATH/$EXP_PATTERN) )
EXP_CONFIG=${EXP_CONFIG_LIST[$SLURM_ARRAY_TASK_ID]}

# check if there is "single_frame" in the EXP_CONFIG
if grep -q "single_frame" $EXP_CONFIG; then
    ARCH_CONFIG=$ARCH_SINGLEFRAME_CONFIG
else
    ARCH_CONFIG=$ARCH_MULTIFRAME_CONFIG
fi

srun python basicsr/train.py --launcher slurm \
    --opt $ARCH_CONFIG \
    --opt $EXP_CONFIG 