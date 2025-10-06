#!/bin/bash
#SBATCH --job-name=cnf
#SBATCH --output=slurm_logs/cnf_%j.out
#SBATCH --error=slurm_logs/cnf_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

# Load conda - use HOME to be machine-independent
source ${HOME}/miniconda3/etc/profile.d/conda.sh

# Activate torch28 environment (used for all experiments)
conda activate torch28


echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

# Execute the driver script and pass along all arguments.
python -u cnf.py "$@"

# Wait a few seconds to ensure the driver has finished writing result_dir.txt.
sleep 10

RESULT_FILE="result_dir.txt"
RESULT_DIR=$(cat "${RESULT_FILE}")
echo "Moving SLURM logs to ${RESULT_DIR}"

# Move the SLURM log files into the results directory.
mv slurm_logs/cnf_${SLURM_JOB_ID}.out ${RESULT_DIR}/slurm_log.out
mv slurm_logs/cnf_${SLURM_JOB_ID}.err ${RESULT_DIR}/slurm_log.err