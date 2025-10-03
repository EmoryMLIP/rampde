#!/bin/bash
#SBATCH --job-name=eval_stl10
#SBATCH --output=slurm_logs/eval_stl10_%j.out
#SBATCH --error=slurm_logs/eval_stl10_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate conda environment
source /local/scratch/lruthot/miniconda3/bin/activate torch26

# Run evaluation script
python evaluate_checkpoints.py "$@"
