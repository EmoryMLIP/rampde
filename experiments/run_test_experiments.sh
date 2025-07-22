#!/bin/bash
# run_test_experiments.sh - Master script for quick test runs of all torchmpnode experiments
# Usage: chmod +x run_test_experiments.sh && ./run_test_experiments.sh
#
# Purpose: Run all experiments with reduced iterations/epochs to:
# 1. Verify all experiments work correctly
# 2. Generate initial timing and performance data  
# 3. Test evaluation scripts while main experiments run
# 4. Get 2-3 validation/CSV rows per experiment for analysis

echo "=========================================="
echo "torchmpnode Test Experiments Runner"
echo "=========================================="
echo "This script runs shortened versions of all experiments to test functionality"
echo "and generate initial performance data."
echo ""

# Check if we're in the experiments directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the experiments/ directory"
    echo "Usage: cd experiments && ./run_test_experiments.sh"
    exit 1
fi

# Create test results directory
mkdir -p results_test
export TEST_RESULTS_DIR="$PWD/results_test"
echo "Test results will be saved to: $TEST_RESULTS_DIR"
echo ""

# Time tracking
start_time=$(date +%s)

echo "Starting torchmpnode test experiment suite at $(date)"
echo ""

# Function to run a test script and track time
run_test() {
    local test_name="$1"
    local script_path="$2"
    local test_start=$(date +%s)
    
    echo "================================================"
    echo "Running $test_name tests..."
    echo "Script: $script_path"
    echo "Started at: $(date)"
    echo "================================================"
    
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        cd "$(dirname "$script_path")"
        ./$(basename "$script_path")
        local exit_code=$?
        cd - > /dev/null
        
        local test_end=$(date +%s)
        local test_duration=$((test_end - test_start))
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ $test_name completed successfully in ${test_duration}s"
        else
            echo "❌ $test_name failed with exit code $exit_code after ${test_duration}s"
        fi
    else
        echo "❌ Test script not found or not executable: $script_path"
    fi
    echo ""
}

# 1. MNIST Test (fastest, good for debugging)
run_test "MNIST Neural ODE" "mnist/run_mnist_test.sh"

# 2. CNF Test (2D toy datasets, quick validation)  
run_test "Continuous Normalizing Flows" "cnf/run_cnf_test.sh"

# 3. STL10 Test (image classification, moderate complexity)
run_test "STL10 Neural ODE" "stl10/run_stl10_test.sh"

# 4. OTFlowLarge Test (most complex, longest even when shortened)
run_test "OTFlow Large-scale" "otflowlarge/run_otflowlarge_test.sh"

# Calculate total runtime
end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "================================================"
echo "All test experiments completed!"
echo "Total runtime: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
echo "================================================"
echo ""

# Wait a moment for all jobs to be submitted
echo "Waiting 10 seconds for all SLURM jobs to be submitted..."
sleep 10

echo "Checking SLURM queue status:"
squeue -u $USER 2>/dev/null || echo "squeue command not available or no jobs in queue"
echo ""

echo "Next steps:"
echo "1. Monitor job progress with: watch -n 30 'squeue -u \$USER'"
echo "2. Check results in: $TEST_RESULTS_DIR"
echo "3. Once jobs complete, analyze results with:"
echo "   cd experiments"
echo "   python collect_results.py --results_dir results_test --output test_results.csv"
echo "   python visualize_results.py test_results.csv --output_dir test_plots"
echo ""

echo "Individual experiment status can be checked with:"
echo "  ls -la mnist/slurm_logs/"
echo "  ls -la cnf/slurm_logs/"
echo "  ls -la stl10/slurm_logs/"  
echo "  ls -la otflowlarge/slurm_logs/"