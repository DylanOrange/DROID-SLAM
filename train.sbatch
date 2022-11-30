#!/bin/bash
#SBATCH --job-name="nocheckpoint_multiple_lr50"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python train.py