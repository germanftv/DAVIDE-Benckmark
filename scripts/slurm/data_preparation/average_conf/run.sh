#!/bin/bash
#SBATCH --job-name=average_confidence
#SBATCH --partition=test
#SBATCH --time=0-00:05:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --output=logs/log.txt
#SBATCH --error=logs/log.txt

# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh

# Define dataset path
DATASET_PATH="${DATASET_ROOT}/test/conf-depth/"

cd ../../../..

python scripts/data_preparation/average_conf.py \
    --root $DATASET_PATH \
    --n_thread 20 \
    --csv ./dataset/annotations/test_avg_conf_depth.csv 
