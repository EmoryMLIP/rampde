#!/bin/bash
# run_mnist.sh - MNIST gradient scaling comparison experiments
# Usage: chmod +x run_mnist.sh ; ./run_mnist.sh

# Default training arguments
default_args=(
  --batch_size  128 
  --nepochs   3  
  --lr        1e-1
  --result_dir ../results_paper
)

# Seed
seed=25

# Make log directory
mkdir -p slurm_logs

echo "Running MNIST Experiments with Gradient Scaling Comparison"
echo "=========================================================="

# Test 1: torchdiffeq and rampde with no scaling in various precisions
echo "Test 1: No scaling comparison - float32, tfloat32, bfloat16"
for precision in "float32" "tfloat32" "bfloat16"; do
  for odeint in "torchdiffeq" "rampde"; do
    fixed_args=(
      --precision "$precision"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --no_grad_scaler
      --no_dynamic_scaler
    )
    echo "Submitting: $odeint $precision no-scaling - ${fixed_args[*]}"
    sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"
  done
done

# Test 2: torchdiffeq in fp16 with and without grad scaling
echo "Test 2: torchdiffeq fp16 scaling comparison"
# torchdiffeq fp16 without grad scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchdiffeq"
  --seed "$seed"
  --no_grad_scaler
)
echo "Submitting: torchdiffeq float16 no-grad-scaler - ${fixed_args[*]}"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"

# torchdiffeq fp16 with grad scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchdiffeq"
  --seed "$seed"
)
echo "Submitting: torchdiffeq float16 with-grad-scaler - ${fixed_args[*]}"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"

# Test 3: rampde in fp16 with different scaling options
echo "Test 3: rampde fp16 scaling comparison"
# rampde fp16 with no scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "rampde"
  --seed "$seed"
  --no_grad_scaler
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 no-scaling - ${fixed_args[*]}"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"

# rampde fp16 with only grad scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "rampde"
  --seed "$seed"
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 only-grad-scaler - ${fixed_args[*]}"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"

# rampde fp16 with only dynamic scaling (default)
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "rampde"
  --seed "$seed"
  --no_grad_scaler
)
echo "Submitting: rampde float16 only-dynamic-scaler - ${fixed_args[*]}"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"

echo "All experiments submitted!"