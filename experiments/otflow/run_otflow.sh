#!/bin/bash
# submit_experiments.sh
# Make sure this file is executable: chmod +x submit_experiments.sh

# Define the parameters.
precisions=( "float32" "float16"  )
methods=("rk4")
odeints=("torchdiffeq" "torchmpnode")
# List of datasets
datasets=("swissroll" )
# datasets=("swissroll" "8gaussians" "pinwheel" "circles" "moons" "2spirals" "checkerboard" "rings")


# FROM OTFlow detailedSetup.md
# python trainToyOTflow.py --data 8gaussians --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,30.0,1 --niters 5000 --lr 1e-1 --val_freq 50  --drop_freq 500 --sample_freq 25 --m 32
# python trainToyOTflow.py --data checkerboard --nt 12 --nt_val 16 --batch_size 10000 --prec double --alph 1.0,15.0,2.0 --niters 20000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32
# python trainToyOTflow.py --data swissroll --nt 8 --nt_val 16 --batch_size 5000 --prec double --alph 1.0,30.0,15.0 --niters 5000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32
# python trainToyOTflow.py --data circles --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,5.0,1.0 --niters 5000 --lr 5e-2 --val_freq 50  --drop_freq 1000 --sample_freq 25 --m 32
# python trainToyOTflow.py --data moons --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,8.0,1.0 --niters 5000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32
# python trainToyOTflow.py --data pinwheel --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,30.0,15.0 --niters 5000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32
# python trainToyOTflow.py --data 2spirals --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,10.0,1.0 --niters 5000 --lr 5e-2 --val_freq 50  --drop_freq 1000 --sample_freq 25 --m 32 


# Map dataset to its specific extra flags
declare -A dataset_args
dataset_args=(
  [8gaussians]="--num_timesteps 8 --num_timesteps_val 12 --num_samples 4096 --alpha 1.0,30.0,0.1 --niters 5000 --lr 1e-1 --test_freq 50 --lr_decay_steps 500 --sample_freq 25"
  [checkerboard]="--num_timesteps 12 --num_timesteps_val 16 --num_samples 8192 --alpha 1.0,15.0,2.0 --niters 20000 --lr 5e-2 --test_freq 50 --lr_decay_steps 1000 --sample_freq 25"
  [swissroll]="--num_timesteps 8 --num_timesteps_val 16 --num_samples 4096 --alpha 1.0,30.0,15.0 --niters 5000 --lr 5e-2 --test_freq 50 --lr_decay_steps 1000 --sample_freq 25"
  [circles]="--num_timesteps 8 --num_timesteps_val 12 --num_samples 4096 --alpha 1.0,8.0,1.0 --niters 5000 --lr 5e-2 --test_freq 50 --lr_decay_steps 1000 --sample_freq 25"
  [moons]="--num_timesteps 8 --num_timesteps_val 12 --num_samples 4096 --alpha 1.0,8.0,1.0 --niters 5000 --lr 5e-2 --test_freq 50 --lr_decay_steps 1000 --sample_freq 25"
  [pinwheel]="--num_timesteps 8 --num_timesteps_val 12 --num_samples 4096 --alpha 1.0,30.0,15.0--niters 5000 --lr 5e-2 --test_freq 50 --lr_decay_steps 1000 --sample_freq 25"
  [2spirals]="--num_timesteps 8 --num_timesteps_val 12 --num_samples 4096 --alpha 1.0,10.0,1.0--niters 5000 --lr 5e-2 --test_freq 50 --lr_decay_steps 1000 --sample_freq 25"
  # Add entries for other datasets as needed
)
# precisions=("float16" "bfloat16" "float32")
# methods=("euler")
# odeints=("torchdiffeq" "torchmpnode")
seed=42

# Create a directory for Slurm log files if needed.
mkdir -p slurm_logs

# Loop over the parameter combinations and submit a job.
for dataset in "${datasets[@]}"; do 
  for precision in "${precisions[@]}"; do
    for method in "${methods[@]}"; do
      for odeint in "${odeints[@]}"; do
        # Construct fixed and dataset‑specific arguments
        fixed_args=(--viz --precision "$precision" --data "$dataset" --method "$method" --odeint "$odeint" --seed "$seed")
        extra_args=${dataset_args[$dataset]}
        echo "Submitting job: ${fixed_args[*]} $extra_args"
        sbatch job_otflow.sbatch "${fixed_args[@]}" $extra_args
      done
    done
  done
done