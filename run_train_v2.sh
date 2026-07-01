#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

enable_lmod

module load container_env pytorch-gpu/2.8.0

crun -p ~/envs/MedNeXt/ python ./train_v2.py --epochs 500 --batch 1 --accum 4 --num_workers 8
