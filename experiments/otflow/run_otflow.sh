#!/bin/bash
# submit_experiments.sh
# Make sure this file is executable: chmod +x submit_experiments.sh

# Define the parameters.
precisions=("float32" "float16")
methods=("rk4")
odeints=( "torchmpnode" "torchdiffeq")
# precisions=("float16" "bfloat16" "float32")
# methods=("euler")
# odeints=("torchdiffeq" "torchmpnode")
seed=42

# Create a directory for Slurm log files if needed.
mkdir -p slurm_logs

# Loop over the parameter combinations and submit a job.
for precision in "${precisions[@]}"; do
  for method in "${methods[@]}"; do
    for odeint in "${odeints[@]}"; do
      echo "Submitting job: precision=$precision, method=$method, odeint=$odeint, seed=$seed"
      sbatch job_otflow.sbatch  --viz  --precision "$precision" --method "$method" --niters 5000 --lr_decay_steps 500 --odeint "$odeint" --seed "$seed"
    done
  done
done