#!/bin/bash
# run_stl10.sh - STL10 gradient scaling comparison experiments
# Usage: chmod +x run_stl10.sh ; ./run_stl10.sh

# Default training arguments
default_args=(
  --batch_size  16 
  --nepochs   120
  --lr 0.05
  --momentum 0.9
  --weight_decay 5e-4
  --width 64
)

# Seed
seed=42

# Make log directory
mkdir -p slurm_logs

echo "Running STL10 Experiments with Gradient Scaling Comparison"
echo "=========================================================="

# Test 1: torchdiffeq and torchmpnode with no scaling in various precisions
echo "Test 1: No scaling comparison - float32, tfloat32, bfloat16"
for precision in "float32" "tfloat32" "bfloat16"; do
  for odeint in "torchdiffeq" "torchmpnode"; do
    fixed_args=(
      --precision "$precision"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --no_grad_scaler
      --no_dynamic_scaler
    )
    echo "Submitting: $odeint $precision no-scaling - ${fixed_args[*]}"
    python ode_stl10.py "${fixed_args[@]}" "${default_args[@]}" &
  done
done

wait  # Wait for all background jobs to complete

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
python ode_stl10.py "${fixed_args[@]}" "${default_args[@]}" &

# torchdiffeq fp16 with grad scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchdiffeq"
  --seed "$seed"
)
echo "Submitting: torchdiffeq float16 with-grad-scaler - ${fixed_args[*]}"
python ode_stl10.py "${fixed_args[@]}" "${default_args[@]}" &

wait  # Wait for all background jobs to complete

# Test 3: torchmpnode in fp16 with different scaling options
echo "Test 3: torchmpnode fp16 scaling comparison"
# torchmpnode fp16 with no scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchmpnode"
  --seed "$seed"
  --no_grad_scaler
  --no_dynamic_scaler
)
echo "Submitting: torchmpnode float16 no-scaling - ${fixed_args[*]}"
python ode_stl10.py "${fixed_args[@]}" "${default_args[@]}" &

# torchmpnode fp16 with only grad scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchmpnode"
  --seed "$seed"
  --no_dynamic_scaler
)
echo "Submitting: torchmpnode float16 only-grad-scaler - ${fixed_args[*]}"
python ode_stl10.py "${fixed_args[@]}" "${default_args[@]}" &

# torchmpnode fp16 with only dynamic scaling (default)
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchmpnode"
  --seed "$seed"
  --no_grad_scaler
)
echo "Submitting: torchmpnode float16 only-dynamic-scaler - ${fixed_args[*]}"
python ode_stl10.py "${fixed_args[@]}" "${default_args[@]}" &

wait  # Wait for all background jobs to complete
echo "All experiments submitted!"