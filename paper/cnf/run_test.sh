#!/bin/bash
# run_cnf_test.sh - CNF test run with reduced iterations for quick validation
# Based on run_cnf.sh but optimized for fast testing
# Usage: chmod +x run_cnf_test.sh && ./run_cnf_test.sh

echo "Running CNF Test Experiments (shortened for quick validation)"
echo "==========================================================="

# Test with only one representative dataset for speed
datasets=("checkerboard" "8gaussians" "2spirals")


# Test dataset arguments - reduced iterations but enough for visualization
declare -A test_dataset_args
test_dataset_args[8gaussians]="--niters 50 --hidden_dim 32 --num_samples 1024 --lr 0.01 --num_timesteps 128 --test_freq 5"
test_dataset_args[checkerboard]="--niters 50 --hidden_dim 32 --num_samples 1024 --lr 0.01 --num_timesteps 128 --test_freq 5"
test_dataset_args[2spirals]="--niters 50 --hidden_dim 32 --num_samples 1024 --lr 0.01 --num_timesteps 128 --test_freq 5"

# Use test results directory
results_dir="./raw_data"
echo "Results will be saved to: $results_dir"

# Seed
seed=24  # Different from production (42) to distinguish test runs

# Make log directory  
mkdir -p slurm_logs

echo "CNF Test Configuration:"
echo "  - Dataset: 8gaussians (representative 2D toy dataset)"
echo "  - Iterations: 200 (vs 2000 in production)"
echo "  - Test frequency: every 20 iterations (10 validation points)"
echo "  - Seed: $seed (vs 42 in production)"
echo "  - Results dir: $results_dir" 
echo ""

# Test 1: Comprehensive precision and scaling comparison
echo "Test 1: Comprehensive precision and scaling comparison"

# float32 tests (no scaling needed)
for dataset in "${datasets[@]}"; do
  for odeint in "torchdiffeq" "rampde"; do
    fixed_args=(
      --precision "float32"
      --data "$dataset"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --results_dir "$results_dir"
      --viz
      --no_grad_scaler
      --no_dynamic_scaler
    )
    extra_args=${test_dataset_args[$dataset]}
    echo "Submitting: $odeint float32 no-scaling - $dataset"
    sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  done
done

# Test different precision types (bfloat16, tfloat32 - no scaling needed)
echo "Testing different precision types:"
for precision in "bfloat16" "tfloat32"; do
  echo "Testing precision: $precision (no scaling needed)"
  for dataset in "${datasets[@]}"; do
    for odeint in "torchdiffeq" "rampde"; do
      fixed_args=(
        --precision "$precision"
        --data "$dataset"
        --method "rk4"
        --odeint "$odeint"
        --seed "$seed"
        --results_dir "$results_dir"
        --viz
        --no_grad_scaler
        --no_dynamic_scaler
      )
      extra_args=${test_dataset_args[$dataset]}
      echo "Submitting: $odeint $precision no-scaling - $dataset"
      sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
    done
  done
done

# float16 tests with various scaling combinations
echo "Float16 scaling combinations:"
for dataset in "${datasets[@]}"; do
  # torchdiffeq float16 + no scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
    --results_dir "$results_dir"
    --viz
    --no_grad_scaler
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchdiffeq float16 no-scaling - $dataset"
  sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  
  # torchdiffeq float16 + grad scaler
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
    --results_dir "$results_dir"
    --viz
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchdiffeq float16 with-grad-scaler - $dataset"
  sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  
  # rampde float16 + no scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --viz
    --no_grad_scaler
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: rampde float16 no-scaling - $dataset"
  sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
  
  # rampde float16 + dynamic scaler only
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --viz
    --no_grad_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: rampde float16 with-dynamic-scaler - $dataset"
  sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
done

# Test 2: Additional rampde float16 scaling strategies
echo ""
echo "Test 2: Additional rampde float16 scaling strategies"

# rampde float16 + grad scaler only
for dataset in "${datasets[@]}"; do
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --viz
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: rampde float16 with-grad-scaler - $dataset"
  sbatch --account=mathg3 job_cnf.sbatch "${fixed_args[@]}" $extra_args
done

echo ""
echo "CNF test experiments submitted!"
echo "Expected runtime: ~5-8 minutes per job"
echo "Expected output: 10 validation points per experiment + visualizations"
echo ""
echo "To generate figures after completion:"
echo "  cd paper/cnf"
echo "  python generate_cnf_overview_wide.py"
echo "  python generate_experiment_table.py --experiment-type cnf --input-dir ./raw_data"
echo ""
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep cnf'"
echo "  tail -f slurm_logs/cnf_*.out"
echo ""
echo "Generated visualizations will be saved in experiment result directories."