#!/bin/bash
#SBATCH --job-name="12_2_val_test"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH --output=/storage/slurm/lud/logs/training/training-%j.out
#SBATCH --error=/storage/slurm/lud/logs/training/training-%j.out
srun python train.py