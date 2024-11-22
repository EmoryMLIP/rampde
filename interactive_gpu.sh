#!/bin/bash
#SBATCH --job-name=interactive_gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=interactive_gpu_%j.log

# Load necessary modules (e.g., CUDA)
module load cuda/12.1

# Activate the virtual environment
source /home/lruthot/scratch/cot-flow/venv/bin/activate

# Start an interactive session
srun --pty /bin/bash