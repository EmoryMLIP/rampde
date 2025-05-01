#!/usr/bin/env bash
# submit_data_loop.sh
# Usage:
#   chmod +x submit_data_loop.sh
#   ./submit_data_loop.sh

# ========== USER CONFIG ==========
# All the choices for your --data argument
datasets=(checkerboard 2spirals pinwheel circles moons rings swissroll 8gaussians)

# Default args (everything except --data)
default_args="\
  --niters 2000\
  --test_freq 20 \
  --num_samples 1024 \
  --num_samples_val 1024 \
  --num_timesteps 128 \
  --width 128 \
  --hidden_dim 32 \
  --gpu 0 \
  --train_dir ./results/cnf \
  --method rk4 \
  --precision float32 \
  --odeint torchmpnode \
  --results_dir ./results/cnf \
  --scaler dynamicscaler \
  --seed 0 \
"

# ========== LOG DIRECTORY ==========
mkdir -p logs

# ========== ACTIVATE ENVIRONMENT ==========
source ~/miniconda3/etc/profile.d/conda.sh   # or your conda path
conda activate torch27

# ========== LOOP OVER DATASETS & RUN ==========
for data in "${datasets[@]}"; do

  logf="logs/cnf_data_${data}.log"
  echo "↪ Running: --data=$data"
  echo "  → logging to $logf"

  python cnf.py \
    --data "$data" \
    $default_args \
    &> "$logf"

  if [ $? -ne 0 ]; then
    echo "FAILED (see $logf)"
  else
    echo "Success."
  fi
  echo "--------------------------------------------------"

done

# ========== CLEANUP ==========
conda deactivate
echo "All runs complete. Logs are in ./logs/"
