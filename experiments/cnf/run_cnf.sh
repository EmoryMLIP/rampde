#!/usr/bin/env bash
# submit_data_loop.sh
# Usage:
#   chmod +x submit_data_loop.sh
#   ./submit_data_loop.sh

# ========== USER CONFIG ==========
# All the choices for your --data argument
datasets=(checkerboard 2spirals pinwheel circles moons rings swissroll 8gaussians)
precisions=( "float32" "float16"  )
methods=("rk4")

# Default args (everything except --data)
default_args="\
  --niters 2000\
  --test_freq 20 \
  --num_samples 1024 \
  --num_samples_val 1024 \
  --num_timesteps 128 \
  --width 128 \
  --hidden_dim 32 \
  --train_dir ./results/cnf \
  --results_dir ./results/cnf \
  --scaler dynamicscaler \
"
seed=42
mkdir -p slurm_logs

for data in "${datasets[@]}"; do
  for precision in "${precisions[@]}"; do
    for method in "${methods[@]}"; do
        fixed_args=(
          --viz
          --precision "$precision"
          --data "$dataset"
          --method "$method"
          --odeint "$odeint"
          --seed "$seed"
        )

        logf="logs/cnf_data_${data}_${precision}_${method}_${odeint}.log"
        echo "↪ Running: --data=$data"
        echo "  → logging to $logf"

        sbatch job_cnf.sbatch "${fixed_args[@]}" "${default_args[@]}" &> "$logf"

        if [ $? -ne 0 ]; then
          echo "FAILED (see $logf)"
        else
          echo "Success."
        fi
        echo "--------------------------------------------------"

    done
  done
done