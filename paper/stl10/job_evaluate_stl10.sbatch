#!/bin/bash
#SBATCH --job-name=eval_stl10
#SBATCH --output=slurm_logs/eval_stl10_%j.out
#SBATCH --error=slurm_logs/eval_stl10_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load conda - use environment variable with default fallback
CONDA_BASE=${CONDA_BASE:-/local/scratch/lruthot/miniconda3}
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate torch28

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

# Run evaluation script
python -u evaluate_checkpoints.py "$@"
