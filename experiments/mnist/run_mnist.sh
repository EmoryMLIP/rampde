#!/bin/bash
# submit_large_experiments_local.sh
# Usage: chmod +x submit_large_experiments_local.sh ; ./submit_large_experiments_local.sh

# Per-dataset arguments
default_args=(
  --batch_size  128 
  --nepochs   5  
  --lr        1e-1
)

# Grid search choices
precisions=( "float16" "bfloat16" "float32" "tfloat32")
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


      logf="logs/ode_mnist_${precision}_${method}_${odeint}.log"
      echo "↪ Running: precision=$precision method=$method odeint=$odeint"

      sbatch job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"

    done
  done
done