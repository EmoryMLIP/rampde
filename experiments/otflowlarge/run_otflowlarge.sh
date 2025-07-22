#!/bin/bash
# run_largeot.sh - Updated with gradient scaling test matrix
# Usage: chmod +x run_largeot.sh ; ./run_largeot.sh

datasets=("bsds300") #"power" "gas" "hepmass" "miniboone" "bsds300"
# datasets=( "hepmass" "miniboone") # Uncomment to run all datasets
# Per-dataset arguments from OT-Flow reference implementation
declare -A dataset_args
dataset_args[power]="--niters 36000 --m 128 --batch_size 10000 --test_batch_size 120000 --lr 0.03 --nt 10 --nt_val 22 --val_freq 30 --weight_decay 0.0 --alph 1.0,500.0,5.0 --drop_freq 0"
dataset_args[gas]="--niters 60000 --m 512 --batch_size 2048 --test_batch_size 55000 --lr 0.01 --nt 10 --nt_val 28 --val_freq 50 --weight_decay 0.0 --alph 1.0,1200.0,40.0 --drop_freq 0 --early_stopping 20"
dataset_args[hepmass]="--niters 40000 --m 256 --batch_size 2048 --test_batch_size 20000 --lr 0.02 --nt 12 --nt_val 24 --val_freq 50 --weight_decay 0.0 --alph 1.0,500.0,40.0 --drop_freq 0 --early_stopping 15"
dataset_args[miniboone]="--niters 8000 --m 256 --batch_size 2048 --test_batch_size 5000 --lr 0.02 --nt 6 --nt_val 10 --val_freq 20 --weight_decay 0.0 --alph 1.0,100.0,15.0 --drop_freq 0 --early_stopping 15"
dataset_args[bsds300]="--niters 10000 --m 1024 --batch_size 512 --test_batch_size 1024 --lr 0.001 --nt 16 --nt_val 30 --val_freq 100 --alph 1.0,2000.0,800.0 --drop_freq 0 --lr_drop 3.3 --early_stopping 15"

# Seed
seed=42

# Make log directory
mkdir -p slurm_logs

echo "Running OTFlow Large Experiments with Gradient Scaling Comparison"
echo "================================================================"
echo "Note: --precision float32 corresponds to --prec single in original OT-Flow"
echo ""

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
      sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
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
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
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
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
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
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
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
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
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
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
done

# Remove wait commands since we're using sbatch instead of background jobs

echo "All training experiments completed!"
echo ""
echo "To run evaluation on trained models, you can use:"
echo "python otflowlarge.py --evaluate --checkpoint_path /path/to/model_checkpoint.pt [other args]"
echo ""
echo "Note: Consider splitting otflowlarge.py into separate train/eval scripts for better workflow."