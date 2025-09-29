#!/bin/bash
# run_mnist_test.sh - MNIST test run with reduced epochs for quick validation
# Based on run_mnist.sh but optimized for fast testing
# Usage: chmod +x run_mnist_test.sh && ./run_mnist_test.sh

echo "Running MNIST Test Experiments (shortened for quick validation)"
echo "=============================================================="

# Test training arguments - significantly reduced for quick runs
test_args=(
  --batch_size  128 
  --nepochs   1        # Reduced from 160 to 1 for quick testing
  --lr        1e-1
  --test_freq 50        # Test every 50 iterations to get 2-3 CSV rows
)

# Use test results directory if available, otherwise use default
results_dir="${TEST_RESULTS_DIR:-./results}"
echo "Results will be saved to: $results_dir"

# Seed
seed=42  # Different from production (25) to distinguish test runs

# Make log directory
mkdir -p slurm_logs

echo "MNIST Test Configuration:"
echo "  - Epochs: 3 (vs 160 in production)"  
echo "  - Test frequency: every epoch (vs every 50 iterations)"
echo "  - Seed: $seed"
echo "  - Results dir: $results_dir"
echo ""

# Test 1: Comprehensive precision and scaling comparison
echo "Test 1: Comprehensive precision and scaling comparison"

# Test different precision types
for precision in "float32" "tfloat32" "bfloat16"; do
  echo "Testing precision: $precision (no scaling needed)"
  for odeint in "torchdiffeq" "rampde"; do
    fixed_args=(
      --precision "$precision"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --results_dir "$results_dir"
      --no_grad_scaler
      --no_dynamic_scaler
    )
    echo "Submitting: $odeint $precision no-scaling"
    sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${test_args[@]}"
  done
done

# float16 tests with various scaling combinations
echo "Float16 scaling combinations:"

# torchdiffeq float16 + no scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchdiffeq"
  --seed "$seed"
  --results_dir "$results_dir"
  --no_grad_scaler
  --no_dynamic_scaler
)
echo "Submitting: torchdiffeq float16 no-scaling"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${test_args[@]}"

# torchdiffeq float16 + grad scaler
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "torchdiffeq"
  --seed "$seed"
  --results_dir "$results_dir"
  --no_dynamic_scaler
)
echo "Submitting: torchdiffeq float16 with-grad-scaler"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${test_args[@]}"

# rampde float16 + no scaling
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "rampde"
  --seed "$seed"
  --results_dir "$results_dir"
  --no_grad_scaler
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 no-scaling"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${test_args[@]}"

# rampde float16 + dynamic scaler only
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "rampde"
  --seed "$seed"
  --results_dir "$results_dir"
  --no_grad_scaler
)
echo "Submitting: rampde float16 with-dynamic-scaler"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${test_args[@]}"

# rampde float16 + grad scaler only
fixed_args=(
  --precision "float16"
  --method "rk4"
  --odeint "rampde"
  --seed "$seed"
  --results_dir "$results_dir"
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 with-grad-scaler"
sbatch --account=mathg3 job_ode_mnist.sbatch "${fixed_args[@]}" "${test_args[@]}"

echo ""
echo "MNIST test experiments submitted!"
echo "Expected runtime: ~5-10 minutes per job"
echo "Expected output: 3-4 CSV rows per experiment (1 per epoch + final)"
echo ""
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep mnist'"
echo "  tail -f slurm_logs/ode_mnist_*.out"