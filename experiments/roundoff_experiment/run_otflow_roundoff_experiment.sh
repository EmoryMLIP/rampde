#!/bin/bash

# Master script to run complete OTFlow roundoff experiment
# Submits individual SLURM jobs for each configuration to avoid JIT conflicts

set -e  # Exit on any error

echo "🚀 OTFlow Roundoff Experiment - Master Script"
echo "=============================================="

# Configuration parameters
PRECISIONS=("float16" "bfloat16")
ODEINT_TYPES=("rampde" "torchdiffeq")
METHODS=("euler" "rk4")
N_TIMESTEPS=(8 16 32 64 128 256)
SEED=42

# Create logs directory
mkdir -p logs

# Function to submit a single job with optional dependency
submit_job() {
    local precision=$1
    local odeint_type=$2
    local scaler_type=$3
    local method=$4
    local n_timesteps=$5
    local seed=$6
    local dependency_job_id=$7  # Optional parameter for job dependency
    
    # Build job name
    local job_name="otflow_${precision}_${odeint_type}_${scaler_type}_${method}_${n_timesteps}"
    
    # Handle scaler_type for environment variable (empty string for None)
    local scaler_env=""
    if [ "$scaler_type" != "None" ]; then
        scaler_env="$scaler_type"
    fi
    
    echo "📤 Submitting: $job_name" >&2
    
    # Submit job with optional dependency
    local job_output
    if [ -n "$dependency_job_id" ]; then
        echo "   └─ Depends on job: $dependency_job_id" >&2
        job_output=$(sbatch \
            --job-name="$job_name" \
            --dependency=afterok:$dependency_job_id \
            --export=PRECISION="$precision",ODEINT_TYPE="$odeint_type",SCALER_TYPE="$scaler_env",METHOD="$method",N_TIMESTEPS="$n_timesteps",SEED="$seed" \
            job_roundoff_single.sbatch 2>&1)
    else
        job_output=$(sbatch \
            --job-name="$job_name" \
            --export=PRECISION="$precision",ODEINT_TYPE="$odeint_type",SCALER_TYPE="$scaler_env",METHOD="$method",N_TIMESTEPS="$n_timesteps",SEED="$seed" \
            job_roundoff_single.sbatch 2>&1)
    fi
    
    # Check if sbatch succeeded
    if [[ $job_output == *"error"* ]]; then
        echo "   ❌ Error: $job_output" >&2
        echo "FAIL"  # Return failure marker
        return 1
    fi
    
    # Extract job ID from output
    local job_id=$(echo "$job_output" | grep -o '[0-9]\+$')
    
    if [ -z "$job_id" ]; then
        echo "   ❌ Could not extract job ID from: $job_output" >&2
        echo "FAIL"
        return 1
    fi
    
    echo "   ✅ Job ID: $job_id" >&2
    echo "$job_id"  # Return the job ID for chaining
}

# Function to get valid scaler types for a given precision and odeint_type
get_scaler_types() {
    local precision=$1
    local odeint_type=$2
    
    if [ "$precision" == "bfloat16" ]; then
        # bfloat16 doesn't need scaling
        echo "None"
    elif [ "$precision" == "float16" ]; then
        if [ "$odeint_type" == "rampde" ]; then
            # rampde supports all scaling types
            echo "none grad dynamic"
        elif [ "$odeint_type" == "torchdiffeq" ]; then
            # torchdiffeq only supports none and grad
            echo "none grad"
        fi
    fi
}

# Generate and submit all valid configurations
total_jobs=0
submitted_jobs=0

echo "🔍 Generating all valid configurations..."

for precision in "${PRECISIONS[@]}"; do
    for odeint_type in "${ODEINT_TYPES[@]}"; do
        # Get valid scaler types for this combination
        scaler_types=$(get_scaler_types "$precision" "$odeint_type")
        
        for scaler_type in $scaler_types; do
            for method in "${METHODS[@]}"; do
                for n_timesteps in "${N_TIMESTEPS[@]}"; do
                    total_jobs=$((total_jobs + 1))
                done
            done
        done
    done
done

echo "📊 Total configurations to run: $total_jobs"
echo ""

# Submit all jobs with sequential dependencies
previous_job_id=""

for precision in "${PRECISIONS[@]}"; do
    for odeint_type in "${ODEINT_TYPES[@]}"; do
        # Get valid scaler types for this combination
        scaler_types=$(get_scaler_types "$precision" "$odeint_type")
        
        for scaler_type in $scaler_types; do
            for method in "${METHODS[@]}"; do
                for n_timesteps in "${N_TIMESTEPS[@]}"; do
                    # Submit job with dependency on previous job (if any)
                    current_job_id=$(submit_job "$precision" "$odeint_type" "$scaler_type" "$method" "$n_timesteps" "$SEED" "$previous_job_id")
                    
                    # Check if job submission failed
                    if [ "$current_job_id" == "FAIL" ]; then
                        echo "❌ Job submission failed, stopping sequential chain"
                        break 4  # Break out of all loops
                    fi
                    
                    submitted_jobs=$((submitted_jobs + 1))
                    
                    # Update previous job ID for next iteration
                    previous_job_id="$current_job_id"
                    
                    # Small delay to avoid overwhelming the scheduler
                    sleep 0.1
                done
            done
        done
    done
done

echo ""
echo "✅ Submitted $submitted_jobs SLURM jobs"
echo ""

# Show queue status
echo "📋 Current queue status:"
squeue -u $USER | head -10

echo ""
echo "🔧 Useful commands:"
echo "   Monitor jobs:     squeue -u $USER"
echo "   Cancel all jobs:  scancel -u $USER"
echo "   Check results:    wc -l results/otflow_roundoff_results.csv"
echo "   View logs:        ls logs/"
echo ""
echo "ℹ️  Jobs are submitted with sequential dependencies to avoid JIT compilation conflicts"
echo "   Only one job will run at a time, ensuring dtype consistency"
echo ""

# Function to monitor progress
monitor_progress() {
    echo "📊 Monitoring progress..."
    while true; do
        # Count running jobs
        running_jobs=$(squeue -u $USER -h | wc -l)
        
        # Count completed results (subtract 1 for header)
        if [ -f "results/otflow_roundoff_results.csv" ]; then
            completed_results=$(($(wc -l < results/otflow_roundoff_results.csv) - 1))
            if [ $completed_results -lt 0 ]; then
                completed_results=0
            fi
        else
            completed_results=0
        fi
        
        # Calculate progress
        remaining_jobs=$running_jobs
        progress_percent=$(( (completed_results * 100) / total_jobs ))
        
        echo "$(date): Progress: $completed_results/$total_jobs ($progress_percent%), Running: $running_jobs (sequential execution)"
        
        # Exit if all jobs are done
        if [ $running_jobs -eq 0 ]; then
            echo "🎉 All jobs completed!"
            break
        fi
        
        # Wait before next check
        sleep 60
    done
}

# Ask user if they want to monitor progress
echo "🤔 Monitor progress? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    monitor_progress
else
    echo "✅ Jobs submitted. Check progress with: squeue -u $USER"
fi

echo ""
echo "🎯 Experiment submitted successfully!"
echo "   Results will be written to: results/otflow_roundoff_results.csv"
echo "   Logs will be in: logs/"