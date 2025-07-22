#!/bin/bash
# run_otflowlarge_test.sh - OTFlow Large test run with drastically reduced iterations
# Based on run_largeot.sh but optimized for fast testing
# Usage: chmod +x run_otflowlarge_test.sh && ./run_otflowlarge_test.sh

echo "Running OTFlow Large Test Experiments (shortened for quick validation)"
echo "===================================================================="

# Use only miniboone dataset for testing (fastest of the available datasets)
datasets=("bsds300")

# Test dataset arguments - drastically reduced iterations for quick testing
declare -A test_dataset_args
# Original miniboone: --niters 8000 --val_freq 20 (400 validation points!)
# Test version: --niters 200 --val_freq 60 (3 validation points: 60, 120, 180)
# test_dataset_args[miniboone]="--niters 200 --m 256 --batch_size 2048 --test_batch_size 5000 --lr 0.02 --nt 6 --nt_val 10 --val_freq 60 --weight_decay 0.0 --alph 1.0,100.0,15.0 --drop_freq 0 --no_early_stopping"
test_dataset_args[bsds300]="--niters 300 --m 1024 --batch_size 512 --test_batch_size 1024 --lr 0.001 --nt 16 --nt_val 30 --val_freq 100 --alph 1.0,2000.0,800.0 --drop_freq 0 --lr_drop 3.3 --early_stopping 15"

# Use test results directory if available
results_dir="${TEST_RESULTS_DIR:-./results/otflowlarge}"
echo "Results will be saved to: $results_dir"

# Seed
seed=23  # Different from production (42) to distinguish test runs

# Make log directory
mkdir -p slurm_logs

echo "OTFlow Large Test Configuration:"
echo "  - Dataset: miniboone only (fastest available dataset)"
echo "  - Iterations: 200 (vs 8000 in production)"
echo "  - Validation frequency: every 60 iterations (3 validation points)"
echo "  - Seed: $seed"
echo "  - Results dir: $results_dir"
echo "  - Early stopping: disabled for consistent test length"
echo ""

# Test 1: Comprehensive precision and scaling comparison
echo "Test 1: Comprehensive precision and scaling comparison"

# float32 tests (no scaling needed)
for dataset in "${datasets[@]}"; do
  for odeint in "torchdiffeq" "torchmpnode"; do
    fixed_args=(
      --precision "float32"
      --data "$dataset"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --results_dir "$results_dir"
      --no_grad_scaler
      --no_dynamic_scaler
    )
    extra_args=${test_dataset_args[$dataset]}
    echo "Submitting: $odeint float32 no-scaling - $dataset"
    sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
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
    --no_grad_scaler
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchdiffeq float16 no-scaling - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchdiffeq float16 + grad scaler
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchdiffeq"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchdiffeq float16 with-grad-scaler - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchmpnode float16 + no scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchmpnode"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchmpnode float16 no-scaling - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchmpnode float16 + dynamic scaler only
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchmpnode"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchmpnode float16 with-dynamic-scaler - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # torchmpnode float16 + grad scaler only
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "torchmpnode"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: torchmpnode float16 with-grad-scaler - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
done

# tfloat32 tests (no scaling needed)
echo "tfloat32 tests:"
for dataset in "${datasets[@]}"; do
  for odeint in "torchdiffeq" "torchmpnode"; do
    fixed_args=(
      --precision "tfloat32"
      --data "$dataset"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --results_dir "$results_dir"
      --no_grad_scaler
      --no_dynamic_scaler
    )
    extra_args=${test_dataset_args[$dataset]}
    echo "Submitting: $odeint tfloat32 no-scaling - $dataset"
    sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  done
done

# bfloat16 tests (no scaling needed - bfloat16 has better numerical stability)
echo "bfloat16 tests:"
for dataset in "${datasets[@]}"; do
  for odeint in "torchdiffeq" "torchmpnode"; do
    fixed_args=(
      --precision "bfloat16"
      --data "$dataset"
      --method "rk4"
      --odeint "$odeint"
      --seed "$seed"
      --results_dir "$results_dir"
      --no_grad_scaler
      --no_dynamic_scaler
    )
    extra_args=${test_dataset_args[$dataset]}
    echo "Submitting: $odeint bfloat16 no-scaling - $dataset"
    sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  done
done

echo ""
echo "All comprehensive test configurations have been submitted above."

echo ""
echo "OTFlow Large test experiments submitted!"
echo "Expected runtime: ~15-25 minutes per job (most complex experiment)"
echo "Expected output: 3 validation points per experiment (iter 60, 120, 180)"
echo ""
echo "Note: This is the longest-running test experiment due to model complexity"
echo ""
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep otflow'"
echo "  tail -f slurm_logs/otflow_*.out"
echo ""
echo "Dataset info:"
echo "  - miniboone: 43,725 samples, 43 dimensions"
echo "  - This tests high-dimensional optimal transport flow learning"