#!/bin/bash
# submit_large_experiments_local.sh
# Usage: chmod +x submit_large_experiments_local.sh ; ./submit_large_experiments_local.sh

datasets=("miniboone") #"power" "gas" "hepmass" "bsds300"

# Per-dataset arguments
declare -A dataset_args
dataset_args[power]="--niters 36000 --alpha 1.0,500.0,5.0 --hidden_dim 128 --num_samples 10000 --lr 0.03 --num_timesteps 10 --num_timesteps_val 22 --num_samples_val 120000 --val_freq 30 --weight_decay 0.0 --drop_freq 0"
dataset_args[gas]="--niters 60000 --alpha 1.0,1200.0,40.0 --hidden_dim 350 --num_samples 2000 --lr 0.01 --num_timesteps 10 --num_timesteps_val 28 --num_samples_val 55000 --val_freq 50 --weight_decay 0.0 --drop_freq 0 --early_stopping 20"
dataset_args[hepmass]="--niters 40000 --alpha 1.0,500.0,40.0 --hidden_dim 256 --num_samples 2000 --lr 0.02 --num_timesteps 12 --num_timesteps_val 24 --num_samples_val 20000 --val_freq 50 --weight_decay 0.0 --drop_freq 0 --early_stopping 15"
dataset_args[miniboone]="--niters 8000 --alpha 1.0,100.0,15.0 --hidden_dim 256 --num_samples 2000 --lr 0.02 --num_timesteps 6 --num_timesteps_val 10 --num_samples_val 5000 --val_freq 20 --weight_decay 0.0 --drop_freq 0 --early_stopping 15"
dataset_args[bsds300]="--niters 120000 --alpha 1.0,2000.0,800.0 --hidden_dim 512 --num_samples 300 --lr 0.001 --num_timesteps 14 --num_timesteps_val 30 --num_samples_val 1000 --val_freq 100 --weight_decay 0.0 --drop_freq 0 --lr_drop 3.3 --early_stopping 15"

# Grid search choices
precisions=("float32" "bfloat16" "float16")
methods=("rk4")
odeints=("torchdiffeq" "torchmpnode")

# Seed
seed=42

# Make log directory
mkdir -p slurm_logs


# Loop!
for dataset in "${datasets[@]}"; do
  for precision in "${precisions[@]}"; do
    for method in "${methods[@]}"; do
      for odeint in "${odeints[@]}"; do

        # assemble the args
        fixed_args=(
          --viz
          --precision "$precision"
          --data "$dataset"
          --method "$method"
          --odeint "$odeint"
          --seed "$seed"
        )
        extra_args=${dataset_args[$dataset]}
        echo "Submitting job: ${fixed_args[*]} $extra_args"
        sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
      done
    done
  done
done
