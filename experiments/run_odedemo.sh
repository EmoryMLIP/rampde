#!/usr/bin/env bash
# submit_ode_demo.sh
# Usage:
#   chmod +x submit_ode_demo.sh
#   ./submit_ode_demo.sh

# ========== USER CONFIG ==========

# Per-dataset arguments
default_args="\
  --data_size    30000 \
  --batch_time   100 \
  --batch_size   20 \
  --niters       2000 \
  --test_freq    10 \
  --gpu          0 \
  --hidden_dim   128 \
  --lr           1e-4 \
"

# Grid search choices
precisions=("float32" "float16" "bfloat16")
methods=("rk4" "euler")
odeints=("torchdiffeq" "torchmpnode")

# ========== LOG DIRECTORY ==========
mkdir -p logs

# ========== ACTIVATE ENVIRONMENT ==========
source ~/miniconda3/etc/profile.d/conda.sh   # adjust if needed
conda activate torch27

# ========== LOOP & RUN ==========

for precision in "${precisions[@]}"; do
  for method in "${methods[@]}"; do
    for odeint in "${odeints[@]}"; do

      fixed_args=(
        --precision "$precision"
        --method    "$method"
        --odeint    "$odeint"
      )
      extra_args="$default_args"

      logf="logs/ode_demo_${precision}_${method}_${odeint}.log"
      echo "↪ Running: precision=$precision method=$method odeint=$odeint"
      echo "  → logging to $logf"

      python ode_demo.py "${fixed_args[@]}" $extra_args &> "$logf"
      if [ $? -ne 0 ]; then
        echo "FAILED (see $logf)"
      else
        echo "Success."
      fi
      echo "--------------------------------------------------"

    done
  done
done


# ========== CLEANUP ==========
conda deactivate
echo "All runs complete. Logs are in ./logs/"
