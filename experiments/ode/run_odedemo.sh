#!/bin/bash
# submit_ode_demo.sh
# Usage:
#   chmod +x submit_ode_demo.sh
#   ./submit_ode_demo.sh

# ========== USER CONFIG ==========

# Per-dataset arguments
default_args="\
  --data_size    30000 \
  --batch_time   100 \
  --batch_size   20 \
  --niters       2000 \
  --test_freq    10 \
  --gpu          0 \
  --hidden_dim   128 \
  --lr           1e-4 \
"

# Grid search choices
precisions=("float32" "float16" "bfloat16")
methods=("rk4" "euler")
odeints=("torchmpnode") #"torchdiffeq" 

mkdir -p slurm_logs


for precision in "${precisions[@]}"; do
  for method in "${methods[@]}"; do
    for odeint in "${odeints[@]}"; do

      fixed_args=(
        --precision "$precision"
        --method    "$method"
        --odeint    "$odeint"
      )
      extra_args="$default_args"

      echo "↪ Running: precision=$precision method=$method odeint=$odeint"

      sbatch job_ode_demo.sbatch "${fixed_args[@]}" $extra_args

    done
  done
done
