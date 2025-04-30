#!/bin/bash
# submit_large_experiments_local.sh
# Usage: chmod +x submit_large_experiments_local.sh ; ./submit_large_experiments_local.sh

# Load conda environment tool
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this path if necessary

# Activate the desired environment
conda activate torch27  # or whatever environment you want

# ========== POWER ==========
echo "Training POWER dataset..."
python otflowlarge.py --data power --niters 36000 --alpha 1.0,500.0,5.0 --hidden_dim 128 --num_samples 10000 --lr 0.03 --num_timesteps 10 --num_timesteps_val 22 --num_samples_val 120000 --val_freq 30 --weight_decay 0.0 --drop_freq 0


# ========== GAS ==========
echo "Training GAS dataset..."
python otflowlarge.py --data gas --niters 60000 --alpha 1.0,1200.0,40.0 --hidden_dim 350 --num_samples 2000 --drop_freq 0 --lr 0.01 --num_timesteps 10 --num_timesteps_val 28 --num_samples_val 55000 --val_freq 50 --weight_decay 0.0 --early_stopping 20



# # ========== HEPMASS ==========
echo "Training HEPMASS dataset..."
python otflowlarge.py --data hepmass --niters 40000 --alpha 1.0,500.0,40.0 --hidden_dim 256 --num_samples 2000 --drop_freq 0 --lr 0.02 --num_timesteps 12 --num_timesteps_val 24 --num_samples_val 20000 --val_freq 50 --weight_decay 0.0 --early_stopping 15


# ========== MINIBOONE ==========
echo "Training MINIBOONE dataset..."
python otflowlarge.py --data miniboone --niters 8000 --alpha 1.0,100.0,15.0 --num_samples 2000 --num_timesteps 6 --num_timesteps_val 10 --lr 0.02 --val_freq 20 --drop_freq 0 --weight_decay 0.0 --hidden_dim 256 --num_samples_val 5000 --early_stopping 15


# ========== BSDS300 ==========
echo "Training BSDS300 dataset..."
python otflowlarge.py --data bsds300 --niters 120000 --alpha 1.0,2000.0,800.0 --num_samples 300 --num_timesteps 14 --num_timesteps_val 30 --lr 0.001 --val_freq 100 --drop_freq 0 --weight_decay 0.0 --hidden_dim 512 --lr_drop 3.3 --num_samples_val 1000  --early_stopping 15

# Deactivate environment after all runs
conda deactivate

echo "✅ All experiments completed."
