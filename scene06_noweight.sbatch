#!/bin/bash
#SBATCH --job-name="training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH --output=/storage/slurm/lud/logs/training/training-%j.out
#SBATCH --error=/storage/slurm/lud/logs/training/training-%j.out
git checkout weight
srun python train.py