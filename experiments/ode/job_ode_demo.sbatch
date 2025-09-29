#!/bin/bash
#SBATCH --job-name=ode_demo
#SBATCH --account=mathg3
#SBATCH --output=slurm_logs/ode_demo_%j.out
#SBATCH --error=slurm_logs/ode_demo_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

# Load your conda environment
source /local/scratch/lruthot/miniconda3/etc/profile.d/conda.sh
conda activate torch27
pip install torchdiffeq

# Determine which conda environment to use based on the --odeint flag
# ODEINT=""
# prev=""
# for arg in "$@"; do
#   if [[ "$prev" == "--odeint" ]]; then
#     ODEINT="$arg"
#     break
#   fi
#   if [[ "$arg" == "--odeint" ]]; then
#     prev="--odeint"
#   else
#     prev=""
#   fi
# done

# if [[ "$ODEINT" == "torchdiffeq" ]]; then
#   conda activate torch26
# elif [[ "$ODEINT" == "rampde" ]]; then
#   conda activate torch27
# else
#   echo "Error: unrecognized odeint value '$ODEINT'" >&2
#   exit 1
# fi


echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

# Execute the driver script and pass along all arguments.
python -u ode_demo.py "$@"

# Wait a few seconds to ensure the driver has finished writing result_dir.txt.
sleep 10

RESULT_FILE="result_dir_${SLURM_JOB_ID}.txt"
RESULT_DIR=$(cat "${RESULT_FILE}")
echo "Moving SLURM logs to ${RESULT_DIR}"

# Move the SLURM log files into the results directory.
# mv slurm_logs/ode_mnist_${SLURM_JOB_ID}.out ${RESULT_DIR}/slurm_log.out
# mv slurm_logs/ode_mnist_${SLURM_JOB_ID}.err ${RESULT_DIR}/slurm_log.err