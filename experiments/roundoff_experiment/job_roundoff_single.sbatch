#!/bin/bash
#SBATCH --job-name=otflow_roundoff
#SBATCH --account=mathg3
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=logs/otflow_%A_%a.out
#SBATCH --error=logs/otflow_%A_%a.err

# SLURM job template for single OTFlow roundoff configuration
# Environment variables expected:
# - PRECISION: float16 or bfloat16
# - ODEINT_TYPE: rampde or torchdiffeq  
# - SCALER_TYPE: none, grad, dynamic, or empty string for None
# - METHOD: euler or rk4
# - N_TIMESTEPS: 8, 16, 32, 64, 128, or 256
# - SEED: random seed (default: 42)

echo "==============================================="
echo "OTFlow Roundoff Experiment - Single Configuration"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Configuration from environment variables
echo ""
echo "Configuration:"
echo "  PRECISION: ${PRECISION}"
echo "  ODEINT_TYPE: ${ODEINT_TYPE}"
echo "  SCALER_TYPE: ${SCALER_TYPE}"
echo "  METHOD: ${METHOD}"
echo "  N_TIMESTEPS: ${N_TIMESTEPS}"
echo "  SEED: ${SEED:-42}"

# Navigate to experiment directory
cd /local/scratch/lruthot/code/rampde/experiments/roundoff_experiment

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate environment
echo ""
echo "Setting up environment..."
source ~/.bashrc
conda activate torch26

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Build command arguments
CMD_ARGS="--precision ${PRECISION} --odeint_type ${ODEINT_TYPE} --method ${METHOD} --n_timesteps ${N_TIMESTEPS}"

# Add scaler_type if not empty
if [ -n "${SCALER_TYPE}" ] && [ "${SCALER_TYPE}" != "None" ]; then
    CMD_ARGS="${CMD_ARGS} --scaler_type ${SCALER_TYPE}"
fi

# Add seed
CMD_ARGS="${CMD_ARGS} --seed ${SEED:-42}"

# Add skip_existing flag for robustness
CMD_ARGS="${CMD_ARGS} --skip_existing"

echo ""
echo "Running command:"
echo "python roundoff_otflow.py ${CMD_ARGS}"
echo ""

# Run the experiment
python roundoff_otflow.py ${CMD_ARGS}

# Capture exit code
EXIT_CODE=$?

echo ""
echo "==============================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "==============================================="

# Check results file
if [ -f "results/otflow_roundoff_results.csv" ]; then
    echo "Results file exists:"
    wc -l results/otflow_roundoff_results.csv
    echo ""
    echo "Last few lines:"
    tail -3 results/otflow_roundoff_results.csv
fi

exit $EXIT_CODE