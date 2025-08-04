#!/bin/bash
#SBATCH --job-name=roundoff_cnf
#SBATCH --output=roundoff_cnf_%j.out
#SBATCH --error=roundoff_cnf_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=zubmain
#SBATCH --gres=gpu:RTX:1
#SBATCH --account=mathg3
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Set up environment
export HOME=/local/scratch/lruthot
cd /local/scratch/lruthot/code/torchmpnode/experiments/roundoff_experiment

# Initialize conda
eval "$(/local/scratch/lruthot/miniconda3/bin/conda shell.bash hook)"
conda activate torch26

# Add torchmpnode to PYTHONPATH
export PYTHONPATH=/local/scratch/lruthot/code/torchmpnode:$PYTHONPATH

# Print environment info
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"

# Run the CNF roundoff experiment
python roundoff_cnf.py

# Print completion message
echo "CNF roundoff experiment completed at $(date)"