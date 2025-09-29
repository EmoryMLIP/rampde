#!/bin/bash
# run_all_experiments.sh - Master script for running all torchmpnode paper experiments
# Usage: chmod +x run_all_experiments.sh && ./run_all_experiments.sh
#
# Purpose: Run comprehensive experiments comparing torchmpnode vs torchdiffeq across:
# - All precision modes: fp32, tfp32, bfp16 for both libraries
# - Float16 with/without grad scaler for both libraries  
# - Float16 with torchmpnode dynamic scaler
# - All major experiment types: MNIST, CNF, STL10, OTFlowLarge

echo "=========================================="
echo "torchmpnode Paper Experiments Runner"
echo "=========================================="
echo "This script runs the complete experiment suite for the torchmpnode paper."
echo "Experiments include all precision configurations across MNIST, CNF, STL10, and OTFlowLarge."
echo ""

# Check if we're in the experiments directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the experiments/ directory"
    echo "Usage: cd experiments && ./run_all_experiments.sh"
    exit 1
fi

# Create results directory
mkdir -p results_paper
export PAPER_RESULTS_DIR="$PWD/results_paper"
echo "Paper results will be saved to: $PAPER_RESULTS_DIR"
echo ""

# Time tracking
start_time=$(date +%s)

echo "Starting torchmpnode paper experiment suite at $(date)"
echo ""

# Function to run an experiment script and track time
run_experiment() {
    local exp_name="$1"
    local script_path="$2"
    local exp_start=$(date +%s)
    
    echo "================================================"
    echo "Running $exp_name experiments..."
    echo "Script: $script_path"
    echo "Started at: $(date)"
    echo "================================================"
    
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        cd "$(dirname "$script_path")"
        ./$(basename "$script_path")
        local exit_code=$?
        cd - > /dev/null
        
        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ $exp_name experiments submitted successfully in ${exp_duration}s"
        else
            echo "❌ $exp_name experiments failed with exit code $exit_code after ${exp_duration}s"
        fi
    else
        echo "❌ Experiment script not found or not executable: $script_path"
    fi
    echo ""
}

# 1. MNIST Neural ODE (fastest, good baseline)
# run_experiment "MNIST Neural ODE" "mnist/run_mnist.sh"

# 2. Continuous Normalizing Flows (2D toy datasets)  
# run_experiment "Continuous Normalizing Flows" "cnf/run_cnf.sh"

# 3. STL10 Neural ODE (image classification)
run_experiment "STL10 Neural ODE" "stl10/run_stl10.sh"

# 4. OTFlowLarge (most complex, runs last as requested)
# run_experiment "OTFlow Large-scale (BSDS300)" "otflowlarge/run_otflowlarge.sh"

# Calculate total runtime
end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "================================================"
echo "All paper experiments submitted!"
echo "Total submission time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
echo "================================================"
echo ""

# Wait a moment for all jobs to be submitted
echo "Waiting 10 seconds for all SLURM jobs to be submitted..."
sleep 10

echo "Checking SLURM queue status:"
squeue -u $USER 2>/dev/null || echo "squeue command not available or no jobs in queue"
echo ""

echo "Paper Experiments Summary:"
echo "========================="
echo "Each experiment runs the following precision configurations:"
echo ""
echo "1. Base precision comparison (no scaling):"
echo "   - torchdiffeq + torchmpnode: float32, tfloat32, bfloat16"
echo ""
echo "2. Float16 scaling comparison:"
echo "   - torchdiffeq: float16 without grad scaler"
echo "   - torchdiffeq: float16 with grad scaler"
echo "   - torchmpnode: float16 without any scaling"
echo "   - torchmpnode: float16 with grad scaler only"
echo "   - torchmpnode: float16 with dynamic scaler only"
echo ""
echo "Experiments include:"
echo "- MNIST: Neural ODE for digit classification"
echo "- CNF: Continuous normalizing flows on 2D toy datasets (checkerboard, 8gaussians, 2spirals)"
echo "- STL10: Neural ODE for image classification"
echo "- OTFlowLarge: Large-scale optimal transport flows (BSDS300 dataset)"
echo ""

echo "Next steps:"
echo "1. Monitor job progress with: watch -n 30 'squeue -u \\$USER'"
echo "2. Check results in: $PAPER_RESULTS_DIR"
echo "3. Once jobs complete, analyze results with:"
echo "   cd experiments"
echo "   python collect_results.py --results_dir results_paper --output paper_results.csv"
echo "   python visualize_results.py paper_results.csv --output_dir paper_plots"
echo ""

echo "Individual experiment status can be checked with:"
echo "  ls -la mnist/slurm_logs/"
echo "  ls -la cnf/slurm_logs/"
echo "  ls -la stl10/slurm_logs/"
echo "  ls -la otflowlarge/slurm_logs/"
echo ""

echo "Total expected jobs: ~78 (6 per MNIST, 54 per CNF, 11 per STL10, 7 per OTFlowLarge)"