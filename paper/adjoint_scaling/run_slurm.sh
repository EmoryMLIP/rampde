#!/bin/bash
#SBATCH --job-name=adjoint_scaling
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=mathg3
#SBATCH --time=00:30:00

# Activate conda environment
source /local/scratch/lruthot/miniconda3/etc/profile.d/conda.sh
conda activate torch26

# Run the experiment
python run_adjoint_scaling.py
