#!/bin/bash
#SBATCH --job-name=otflow
#SBATCH --output=slurm_logs/otflow_%j.out
#SBATCH --error=slurm_logs/otflow_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

# Load your conda environment
source /local/scratch/lruthot/miniconda3/etc/profile.d/conda.sh
conda activate torch26

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

# Execute the driver script and pass along all arguments.
python -u otflow_exp.py "$@"

# Wait a few seconds to ensure the driver has finished writing result_dir.txt.
sleep 10

RESULT_FILE="result_dir_${SLURM_JOB_ID}.txt"
RESULT_DIR=$(cat "${RESULT_FILE}")
echo "Moving SLURM logs to ${RESULT_DIR}"

# Move the SLURM log files into the results directory.
# mv slurm_logs/otflow_${SLURM_JOB_ID}.out ${RESULT_DIR}/slurm_log.out
# mv slurm_logs/otflow_${SLURM_JOB_ID}.err ${RESULT_DIR}/slurm_log.err