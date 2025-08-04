#!/bin/bash
#SBATCH --job-name=roundoff_exp
#SBATCH --output=slurm_logs/roundoff_%j.out
#SBATCH --error=slurm_logs/roundoff_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=mathg3
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Load environment
source /local/scratch/lruthot/miniconda3/bin/activate torch26

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to experiment directory
cd /local/scratch/lruthot/code/torchmpnode/experiments/roundoff_experiment

# Print environment info
echo "Starting roundoff experiments..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run the experiments
./run_roundoff.sh

echo "Experiments completed!"