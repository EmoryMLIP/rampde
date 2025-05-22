#!/bin/bash
# submit_large_experiments_local.sh
# Usage: chmod +x submit_large_experiments_local.sh ; ./submit_large_experiments_local.sh

# Per-dataset arguments
default_args=(
  --batch_size  64 
  --nepochs   120 
  --lr 5e-2
  --weight_decay 1e-4
)

# Grid search choices
precisions=( "float32")
methods=("rk4")
odeints=( "torchmpnode")


mkdir -p slurm_logs


for precision in "${precisions[@]}"; do
  for method in "${methods[@]}"; do
    for odeint in "${odeints[@]}"; do

      fixed_args=(
        --precision "$precision"
        --method    "$method"
        --odeint    "$odeint"
      )


      logf="logs/ode_stl10_${precision}_${method}_${odeint}.log"
      echo "↪ Running: precision=$precision method=$method odeint=$odeint"

      sbatch job_ode_stl10.sbatch "${fixed_args[@]}" "${default_args[@]}"

    done
  done
done