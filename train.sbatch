#!/bin/bash
#SBATCH --job-name="depth_0001_0_0_4_4"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=6G
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python train.py