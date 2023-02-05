#!/bin/bash
#SBATCH --job-name="30*101_15_2_objectflow"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=6G
#SBATCH --time=18:00:00
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
git checkout 30-101_object
srun python train.py