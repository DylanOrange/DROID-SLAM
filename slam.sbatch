#!/bin/bash
#SBATCH --job-name="SLAM"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:16G
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH --output=/home/stud/lud/DROID-SLAM/logs/slam.out
#SBATCH --error=/home/stud/lud/DROID-SLAM/logs/slam.out
srun python evaluation_scripts/test_vkitti2.py