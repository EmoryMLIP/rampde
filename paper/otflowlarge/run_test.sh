#!/bin/bash
# run_otflowlarge_test.sh - OTFlow Large test run with drastically reduced iterations
# Based on run_largeot.sh but optimized for fast testing
# Usage: chmod +x run_otflowlarge_test.sh && ./run_otflowlarge_test.sh

echo "Running OTFlow Large Test Experiments (shortened for quick validation)"
echo "===================================================================="

# Use only bsds300 dataset for testing
datasets=("bsds300")

# Test dataset arguments - reduced iterations but enough for table generation
declare -A test_dataset_args
# Original: --niters 8000+ with frequent validation
# Test version: --niters 500 with val_freq 50 (10 validation points)
test_dataset_args[bsds300]="--niters 500 --m 1024 --batch_size 512 --test_batch_size 1024 --lr 0.001 --nt 16 --nt_val 30 --val_freq 50 --alph 1.0,2000.0,800.0 --drop_freq 0 --lr_drop 3.3 --early_stopping 15"

# Use test results directory
results_dir="./raw_data"
echo "Results will be saved to: $results_dir"

# Seed
seed=23  # Different from production (42) to distinguish test runs

# Make log directory
mkdir -p slurm_logs

echo "OTFlow Large Test Configuration:"
echo "  - Dataset: bsds300 only"
echo "  - Iterations: 500 (vs 5000+ in production)"
echo "  - Validation frequency: every 50 iterations (10 validation points)"
echo "  - Seed: $seed (vs 42 in production)"
echo "  - Results dir: $results_dir"
echo ""

# Check if BSDS300 dataset exists, if not, prompt to download
if [ ! -f "data/BSDS300/BSDS300.hdf5" ]; then
    echo "ERROR: BSDS300 dataset not found!"
    echo "Please run: python download_datasets.py"
    exit 1
fi

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
  
  # rampde float16 + no scaling
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: rampde float16 no-scaling - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # rampde float16 + dynamic scaler only
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_grad_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: rampde float16 with-dynamic-scaler - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
  
  # rampde float16 + grad scaler only
  fixed_args=(
    --precision "float16"
    --data "$dataset"
    --method "rk4"
    --odeint "rampde"
    --seed "$seed"
    --results_dir "$results_dir"
    --no_dynamic_scaler
  )
  extra_args=${test_dataset_args[$dataset]}
  echo "Submitting: rampde float16 with-grad-scaler - $dataset"
  sbatch --account=mathg3 job_otflowlarge.sbatch "${fixed_args[@]}" $extra_args
done

# tfloat32 tests (no scaling needed)
echo "tfloat32 tests:"
for dataset in "${datasets[@]}"; do
  for odeint in "torchdiffeq" "rampde"; do
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
  for odeint in "torchdiffeq" "rampde"; do
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
echo "Expected runtime: ~15-25 minutes per job"
echo "Expected output: 10 validation points per experiment"
echo ""
echo "To generate tables after completion:"
echo "  cd paper/otflowlarge"
echo "  python generate_otflowlarge_table.py"
echo ""
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep otflow'"
echo "  tail -f slurm_logs/otflow_*.out"