#!/bin/bash
# run_largeot.sh - Updated with gradient scaling test matrix
# Usage: chmod +x run_largeot.sh ; ./run_largeot.sh

datasets=("bsds300") #"power" "gas" "hepmass" "miniboone" "bsds300"

# Per-dataset arguments from OT-Flow detailedSetup.md
declare -A dataset_args
dataset_args[power]="--niters 36000 --hidden_dim 128 --num_samples 10000 --num_samples_val 120000 --lr 0.03 --num_timesteps 10 --num_timesteps_val 22 --test_freq 30 --weight_decay 0.0 --alpha 1.0,500.0,5.0"
dataset_args[gas]="--niters 60000 --hidden_dim 350 --num_samples 2000 --num_samples_val 55000 --lr 0.01 --num_timesteps 10 --num_timesteps_val 28 --test_freq 50 --weight_decay 0.0 --alpha 1.0,1200.0,40.0"
dataset_args[hepmass]="--niters 40000 --hidden_dim 256 --num_samples 2000 --num_samples_val 20000 --lr 0.02 --num_timesteps 12 --num_timesteps_val 24 --test_freq 50 --weight_decay 0.0 --alpha 1.0,500.0,40.0"
dataset_args[miniboone]="--niters 8000 --hidden_dim 256 --num_samples 2000 --lr 0.02 --num_timesteps 6 --num_timesteps_val 10 --test_freq 20 --weight_decay 0.0 --alpha 1.0,100.0,15.0"
dataset_args[bsds300]="--niters 5000 --hidden_dim 256 --num_samples 200 --lr 0.001 --num_timesteps 8 --test_freq 50 --alpha 1.0,2000.0,800.0"

# Seed
seed=42

# Make log directory
mkdir -p slurm_logs

echo "Running OTFlow Large Experiments with Gradient Scaling Comparison"
echo "================================================================"

# Test 1: torchdiffeq and torchmpnode with no scaling in various precisions
echo "Test 1: No scaling comparison - float32, tfloat32, bfloat16"
for dataset in "${datasets[@]}"; do
  for precision in "float32" "tfloat32" "bfloat16"; do
    for odeint in "torchdiffeq" "torchmpnode"; do
      fixed_args=(
        --precision "$precision"
        --data "$dataset"
        --method "rk4"
        --odeint "$odeint"
        --seed "$seed"
        --no_grad_scaler
        --no_dynamic_scaler
      )
      extra_args=${dataset_args[$dataset]}
      echo "Submitting: $odeint $precision no-scaling - ${fixed_args[*]} $extra_args"
      sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
    done
  done
done

# Remove wait commands since we're using sbatch instead of background jobs

# Test 2: torchdiffeq in fp16 with and without grad scaling
echo "Test 2: torchdiffeq fp16 scaling comparison"
for dataset in "${datasets[@]}"; do
  # torchdiffeq fp16 without grad scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
    --no_grad_scaler
  )
  extra_args=${dataset_args[$dataset]}
  echo "Submitting: torchdiffeq float16 no-grad-scaler - ${fixed_args[*]} $extra_args"
  sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchdiffeq fp16 with grad scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
  )
  extra_args=${dataset_args[$dataset]}
  echo "Submitting: torchdiffeq float16 with-grad-scaler - ${fixed_args[*]} $extra_args"
  sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
done

# Remove wait commands since we're using sbatch instead of background jobs

# Test 3: torchmpnode in fp16 with different scaling options
echo "Test 3: torchmpnode fp16 scaling comparison"
for dataset in "${datasets[@]}"; do
  # torchmpnode fp16 with no scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchmpnode"
    --seed "$seed"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  extra_args=${dataset_args[$dataset]}
  echo "Submitting: torchmpnode float16 no-scaling - ${fixed_args[*]} $extra_args"
  sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchmpnode fp16 with only grad scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchmpnode"
    --seed "$seed"
    --no_dynamic_scaler
  )
  extra_args=${dataset_args[$dataset]}
  echo "Submitting: torchmpnode float16 only-grad-scaler - ${fixed_args[*]} $extra_args"
  sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchmpnode fp16 with only dynamic scaling (default)
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchmpnode"
    --seed "$seed"
    --no_grad_scaler
  )
  extra_args=${dataset_args[$dataset]}
  echo "Submitting: torchmpnode float16 only-dynamic-scaler - ${fixed_args[*]} $extra_args"
  sbatch job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
done

# Remove wait commands since we're using sbatch instead of background jobs

echo "All training experiments completed!"
echo ""
echo "To run evaluation on trained models, you can use:"
echo "python otflowlarge.py --evaluate --checkpoint_path /path/to/model_checkpoint.pt [other args]"
echo ""
echo "Note: Consider splitting otflowlarge.py into separate train/eval scripts for better workflow."