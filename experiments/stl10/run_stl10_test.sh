#!/bin/bash
# run_stl10_test.sh - STL10 test run with minimal epochs for quick validation  
# Based on run_stl10.sh but optimized for fast testing
# Usage: chmod +x run_stl10_test.sh && ./run_stl10_test.sh

echo "Running STL10 Test Experiments (shortened for quick validation)"
echo "============================================================="

# Test training arguments - kept minimal for quick testing
test_args=(
  --batch_size  16      # Keep small batch size for memory efficiency
  --nepochs   2         # Slightly increased from 2 to get better validation
  --lr 0.05
  --momentum 0.9
  --weight_decay 5e-4
  --test_freq 1         # Test every epoch to get 3 CSV rows
  --width 128
)

# Use test results directory if available
results_dir="${TEST_RESULTS_DIR:-./results}"
echo "Results will be saved to: $results_dir"

# Seed  
seed=26  # Different from production (25) to distinguish test runs

# Make log directory
mkdir -p slurm_logs

echo "STL10 Test Configuration:"
echo "  - Epochs: 3 (vs 160 in typical production runs)"
echo "  - Batch size: 16 (memory efficient for testing)"
echo "  - Test frequency: every epoch (3 validation points)"
echo "  - Seed: $seed"
echo "  - Results dir: $results_dir"
echo ""

# Test 1: Comprehensive precision and scaling comparison
echo "Test 1: Comprehensive precision and scaling comparison"

# float32 tests (no scaling needed)
for odeint in "torchdiffeq" "rampde"; do
  fixed_args=(
    --precision "float32"
    --method "rk4"
    --odeint "$odeint"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  echo "Submitting: $odeint float32 no-scaling"
  sbatch --account=mathg3 job_ode_stl10.sbatch "${fixed_args[@]}" "${test_args[@]}"
done

# Test different precision types  
echo "Testing different precision types:"
for precision in "float16" "bfloat16" "tfloat32"; do
  echo "Testing precision: $precision"
  
  # torchdiffeq precision + no scaling
  fixed_args=(
    --precision "$precision"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  echo "Submitting: torchdiffeq $precision no-scaling"
  sbatch --account=mathg3 job_ode_stl10.sbatch "${fixed_args[@]}" "${test_args[@]}"

  # torchdiffeq precision + grad scaler
  fixed_args=(
    --precision "$precision"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_dynamic_scaler
  )
  echo "Submitting: torchdiffeq $precision with-grad-scaler"
  sbatch --account=mathg3 job_ode_stl10.sbatch "${fixed_args[@]}" "${test_args[@]}"

  # rampde precision + no scaling
  fixed_args=(
    --precision "$precision"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  echo "Submitting: rampde $precision no-scaling"
  sbatch --account=mathg3 job_ode_stl10.sbatch "${fixed_args[@]}" "${test_args[@]}"

  # rampde precision + dynamic scaler only
  fixed_args=(
    --precision "$precision"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
  )
  echo "Submitting: rampde $precision with-dynamic-scaler"
  sbatch --account=mathg3 job_ode_stl10.sbatch "${fixed_args[@]}" "${test_args[@]}"

  # rampde precision + grad scaler only
  fixed_args=(
    --precision "$precision"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_dynamic_scaler
  )
  echo "Submitting: rampde $precision with-grad-scaler"
  sbatch --account=mathg3 job_ode_stl10.sbatch "${fixed_args[@]}" "${test_args[@]}"
done

echo ""
echo "STL10 test experiments submitted!"
echo "Expected runtime: ~10-15 minutes per job (depends on dataset download)"
echo "Expected output: 3-4 CSV rows per experiment (1 per epoch + final)"
echo ""
echo "Note: First run may take longer due to STL10 dataset download"
echo ""
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep stl10'"
echo "  tail -f slurm_logs/ode_stl10_*.out"