#!/bin/bash
# run_roundoff_test.sh - Roundoff test run with reduced iterations
# Usage: chmod +x run_test.sh && ./run_test.sh

echo "Running Roundoff Error Analysis Test (shortened for quick validation)"
echo "====================================================================="

# Activate conda environment
source ~/.bashrc
conda activate torch26

# Add rampde to Python path
export PYTHONPATH=/local/scratch/lruthot/code/rampde:$PYTHONPATH

# Test configuration
results_dir="./raw_data"
seed=999  # Different from production (42) to distinguish test runs

echo "Roundoff Test Configuration:"
echo "  - Iterations: 300 (vs 2000 in production)"
echo "  - Dataset: 8gaussians"
echo "  - Seed: $seed (vs 42 in production)"
echo "  - Results dir: $results_dir"
echo ""

# Run minimal test (reduced iterations but enough for plotting)
# 300 iterations gives enough data points to show error trends
python roundoff_cnf.py --niters 300 --test_seed $seed --results_dir $results_dir

echo ""
echo "Roundoff test experiment completed!"
echo "Expected runtime: ~8-10 minutes"
echo "Expected output: CSV with error metrics across iterations"
echo ""
echo "To generate figures after completion:"
echo "  cd paper/roundoff"
echo "  python plot_cnf_roundoff.py --create-combined"