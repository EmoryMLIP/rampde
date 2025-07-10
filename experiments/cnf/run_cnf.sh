#!/bin/bash
# submit_data_loop.sh
# Usage:
#   chmod +x submit_data_loop.sh
#   ./submit_data_loop.sh

# ========== USER CONFIG ==========
# All the choices for your --data argument
datasets=("pinwheel" "2spirals")
precisions=( "bfloat16" "float16" "float32"  )
methods=("rk4")
odeints=("torchdiffeq" "torchmpnode") # "torchdiffeq" 
num_timesteps_list=(32 64 128 )

# Default args (everything except --data and --num_timesteps)
# CORRECT: an array, so each flag and value is its own element
# Note: removed --train_dir and --results_dir as they are automatically handled by setup_experiment()
default_args=(
  --niters 2000
  --test_freq 20
  --num_samples 1024
  --num_samples_val 1024
  --width 128
  --hidden_dim 32
  --scaler dynamicscaler
)

seed=42
mkdir -p slurm_logs

for data in "${datasets[@]}"; do
  for precision in "${precisions[@]}"; do
    for method in "${methods[@]}"; do
      for odeint in "${odeints[@]}"; do
        for num_timesteps in "${num_timesteps_list[@]}"; do
          fixed_args=(
            --viz
            --precision "$precision"
            --data "$data"
            --method "$method"
            --odeint "$odeint"
            --num_timesteps "$num_timesteps"
            --seed "$seed"
          )
          echo "↪ Running: --data=$data --precision=$precision --method=$method --odeint=$odeint --num_timesteps=$num_timesteps"
          sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" "${default_args[@]}"
          sleep 2
        done
      done
    done
  done
done
