#!/bin/bash
#SBATCH --job-name="training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2,VRAM:48G
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH --output=/storage/slurm/lud/logs/training/training-%j.out
#SBATCH --error=/storage/slurm/lud/logs/training/training-%j.out
srun python train.py