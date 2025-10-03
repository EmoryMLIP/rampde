#!/bin/bash
# run_cnf.sh - CNF gradient scaling comparison experiments
# Usage: chmod +x run_cnf.sh ; ./run_cnf.sh

datasets=("checkerboard" "8gaussians" "2spirals")

# Per-dataset arguments - simplified for comparison tests
declare -A dataset_args
dataset_args[checkerboard]="--niters 2000 --hidden_dim 32 --num_samples 1024 --lr 0.01 --num_timesteps 128 --test_freq 20"
dataset_args[8gaussians]="--niters 2000 --hidden_dim 32 --num_samples 1024 --lr 0.01 --num_timesteps 128 --test_freq 20"
dataset_args[2spirals]="--niters 2000 --hidden_dim 32 --num_samples 1024 --lr 0.01 --num_timesteps 128 --test_freq 20"

# Seed
seed=42

# Make log directory
mkdir -p slurm_logs

echo "Running CNF Experiments with Gradient Scaling Comparison"
echo "======================================================="

# Test 1: torchdiffeq and rampde with no scaling in various precisions
echo "Test 1: No scaling comparison - float32, tfloat32, bfloat16"
for dataset in "${datasets[@]}"; do
  for precision in "tfloat32"; do
  # for precision in "float32" "tfloat32" "bfloat16"; do
    for odeint in "torchdiffeq" "rampde"; do
      fixed_args=(
        --precision "$precision"
        --data "$dataset"
        --method "rk4"
        --odeint "$odeint"
        --seed "$seed"
        --no_grad_scaler
        --no_dynamic_scaler
        --viz
        --results_dir ./raw_data
      )
      extra_args=${dataset_args[$dataset]}
      echo "Submitting: $odeint $precision no-scaling - ${fixed_args[*]} $extra_args"
      sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
    done
  done
done

# # Remove wait commands since we're using sbatch instead of background jobs

# # Test 2: torchdiffeq in fp16 with and without grad scaling
# echo "Test 2: torchdiffeq fp16 scaling comparison"
# for dataset in "${datasets[@]}"; do
#   # torchdiffeq fp16 without grad scaling
#   fixed_args=(
#     --precision "float16"
#     --data "$dataset"
#     --method "rk4"
#     --odeint "torchdiffeq"
#     --seed "$seed"
#     --no_grad_scaler
#     --viz
#     --results_dir ../results_paper_cnf
#   )
#   extra_args=${dataset_args[$dataset]}
#   echo "Submitting: torchdiffeq float16 no-grad-scaler - ${fixed_args[*]} $extra_args"
#   sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  
#   # torchdiffeq fp16 with grad scaling
#   fixed_args=(
#     --precision "float16"
#     --data "$dataset"
#     --method "rk4"
#     --odeint "torchdiffeq"
#     --seed "$seed"
#     --viz
#     --results_dir ../results_paper_cnf
#   )
#   extra_args=${dataset_args[$dataset]}
#   echo "Submitting: torchdiffeq float16 with-grad-scaler - ${fixed_args[*]} $extra_args"
#   sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
# done

# # Remove wait commands since we're using sbatch instead of background jobs

# # Test 3: rampde in fp16 with different scaling options
# echo "Test 3: rampde fp16 scaling comparison"
# for dataset in "${datasets[@]}"; do
#   # rampde fp16 with no scaling
#   fixed_args=(
#     --precision "float16"
#     --data "$dataset"
#     --method "rk4"
#     --odeint "rampde"
#     --seed "$seed"
#     --no_grad_scaler
#     --no_dynamic_scaler
#     --viz
#     --results_dir ../results_paper_cnf
#   )
#   extra_args=${dataset_args[$dataset]}
#   echo "Submitting: rampde float16 no-scaling - ${fixed_args[*]} $extra_args"
#   sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  
#   # rampde fp16 with only grad scaling
#   fixed_args=(
#     --precision "float16"
#     --data "$dataset"
#     --method "rk4"
#     --odeint "rampde"
#     --seed "$seed"
#     --no_dynamic_scaler
#     --viz
#     --results_dir ../results_paper_cnf
#   )
#   extra_args=${dataset_args[$dataset]}
#   echo "Submitting: rampde float16 only-grad-scaler - ${fixed_args[*]} $extra_args"
#   sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  
#   # rampde fp16 with only dynamic scaling (default)
#   fixed_args=(
#     --precision "float16"
#     --data "$dataset"
#     --method "rk4"
#     --odeint "rampde"
#     --seed "$seed"
#     --no_grad_scaler
#     --viz
#     --results_dir ../results_paper_cnf
#   )
#   extra_args=${dataset_args[$dataset]}
#   echo "Submitting: rampde float16 only-dynamic-scaler - ${fixed_args[*]} $extra_args"
#   sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
# done

# Remove wait commands since we're using sbatch instead of background jobs
echo "All experiments submitted!"
