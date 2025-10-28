#!/bin/bash
#SBATCH --job-name=otflow
#SBATCH --output=slurm_logs/otflow_%j.out
#SBATCH --error=slurm_logs/otflow_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

# Load conda - use environment variable with default fallback
CONDA_BASE=${CONDA_BASE:-/local/scratch/lruthot/miniconda3}
source ${CONDA_BASE}/etc/profile.d/conda.sh

# Activate torch28 environment (used for all experiments)
conda activate torch28

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

# Execute the driver script and pass along all arguments.
python -u otflowlarge.py "$@"

# Wait a few seconds to ensure the driver has finished writing result_dir.txt.
sleep 10

RESULT_FILE="result_dir_${SLURM_JOB_ID}.txt"
RESULT_DIR=$(cat "${RESULT_FILE}")
echo "Moving SLURM logs to ${RESULT_DIR}"

# Move the SLURM log files into the results directory.
# mv slurm_logs/otflow_${SLURM_JOB_ID}.out ${RESULT_DIR}/slurm_log.out
# mv slurm_logs/otflow_${SLURM_JOB_ID}.err ${RESULT_DIR}/slurm_log.err