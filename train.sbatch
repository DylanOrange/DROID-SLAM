#!/bin/bash
#SBATCH --job-name="12_2_randomselect"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python train.py