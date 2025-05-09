#!/bin/bash
#SBATCH --job-name=lmdb_generation
#SBATCH --partition=small
#SBATCH --time=0-6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --nodes=1
#SBATCH --output=logs/log.txt
#SBATCH --error=logs/log.txt

# Activate enviroment, export variables
source ~/env_vars/DAVIDE.sh


cd ../../../..

python scripts/data_preparation/create_lmdb.py \
    --dataset_path $DATASET_ROOT \
    --n_thread 20 \