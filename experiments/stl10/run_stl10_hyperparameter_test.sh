#!/bin/bash
# run_stl10_hyperparameter_test.sh - STL10 hyperparameter sweep with rampde bfloat16
# Usage: chmod +x run_stl10_hyperparameter_test.sh ; ./run_stl10_hyperparameter_test.sh

# Default training arguments (modified for hyperparameter testing)
default_args=(
  --batch_size  16 
  --lr 0.05
  --momentum 0.9
  --width 64
  --results_dir 2025-07-29-hyperparameter_test
  --precision "bfloat16"
  --method "rk4"
  --odeint "rampde"
  --no_grad_scaler  # Use rampde's native mixed precision
)

# Fixed configuration for all experiments
fixed_args=(
  --precision "bfloat16"
  --method "rk4"
  --odeint "rampde"
  --no_grad_scaler  # rampde handles mixed precision internally
)

# Hyperparameter sweep arrays
weight_decays=("1e-5" "5e-5" "1e-4")
iterations=("80" "120" "160")

# Seed for reproducibility
seed=25

# Make log directory
mkdir -p slurm_logs

echo "Running STL10 Hyperparameter Sweep with rampde bfloat16"
echo "============================================================="
echo "Weight decay values: ${weight_decays[*]}"
echo "Iteration counts: ${iterations[*]}"
echo "Results will be saved to: 2025-07-29-hyperparameter_test/"
echo ""

# Counter for experiment tracking
experiment_count=0
total_experiments=$((${#weight_decays[@]} * ${#iterations[@]}))

# Hyperparameter sweep
for nepochs in "${iterations[@]}"; do
  for weight_decay in "${weight_decays[@]}"; do
    experiment_count=$((experiment_count + 1))
    
    # Combine all arguments
    experiment_args=(
      "${fixed_args[@]}"
      --weight_decay "$weight_decay"
      --nepochs "$nepochs"
      --seed "$seed"
      "${default_args[@]}"
    )
    
    echo "[$experiment_count/$total_experiments] Submitting: weight_decay=$weight_decay, nepochs=$nepochs"
    echo "  Command: sbatch --account=mathg3 job_ode_stl10.sbatch ${experiment_args[*]}"
    
    # Submit the job
    sbatch --account=mathg3 job_ode_stl10.sbatch "${experiment_args[@]}"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 2
  done
done

echo ""
echo "All $total_experiments experiments submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: 2025-07-29-hyperparameter_test/"
echo ""
echo "Experiment matrix:"
echo "=================="
printf "%-15s | %-10s\n" "Weight Decay" "Iterations"
echo "----------------+-----------"
for weight_decay in "${weight_decays[@]}"; do
  for nepochs in "${iterations[@]}"; do
    printf "%-15s | %-10s\n" "$weight_decay" "$nepochs"
  done
done