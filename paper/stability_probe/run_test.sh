#!/bin/bash
# Quick validation: two alpha values with reduced steps.

set -euo pipefail
cd "$(dirname "$0")"

source /local/scratch/lruthot/miniconda3/etc/profile.d/conda.sh
conda activate torch28

echo "=== Stability probe: quick test ==="
python stability_probe.py \
    --n-steps 100 \
    --n-steps-grad 50 \
    --alphas 0.5 1.01 \
    --batch-size 2 \
    --seed 999 \
    --results-dir raw_data_test
