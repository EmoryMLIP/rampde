#!/bin/bash
#
# Run all roundoff experiments
# This script runs lightweight roundoff error analysis for CNF and OTFlow
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate conda environment
echo "Activating torch26 environment..."
source /local/scratch/lruthot/miniconda3/bin/activate torch26

# Create results directory
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Function to run an experiment
run_experiment() {
    local experiment_name=$1
    local script_name=$2
    
    echo ""
    echo "========================================"
    echo "Running ${experiment_name} roundoff experiment..."
    echo "========================================"
    
    python "${SCRIPT_DIR}/${script_name}" 2>&1 | tee "${RESULTS_DIR}/${experiment_name}_log.txt"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ ${experiment_name} completed successfully"
    else
        echo "✗ ${experiment_name} failed"
        exit 1
    fi
}

# Record start time
START_TIME=$(date +%s)
echo "Starting roundoff experiments at $(date)"
echo "Results will be saved to: ${RESULTS_DIR}"

# Run experiments
run_experiment "CNF" "roundoff_cnf.py"
run_experiment "OTFlow" "roundoff_otflow.py"

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Total runtime: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
echo "Results saved in: ${RESULTS_DIR}"
echo ""
echo "To visualize results, run:"
echo "  python ${SCRIPT_DIR}/plot_roundoff.py"
echo "========================================"