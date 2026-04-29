#!/bin/bash
# Full stability probe sweep on the trained STL-10 ode2 block (~30 min on GPU).
# fp64 integration dominates runtime (~50x slower than fp32 due to lack of
# tensor-core support).

set -euo pipefail
cd "$(dirname "$0")"

source /local/scratch/lruthot/miniconda3/etc/profile.d/conda.sh
conda activate torch28

echo "=== Stability probe: full experiment ==="
# Alpha grid (9 points):
#   0.5, 0.99   stable regime; 0.99 is "just below Dahlquist" and exposes
#               the bf16 accumulation floor on the linearized block.
#   1.0         Dahlquist boundary (linearized blowup point).
#   2.0..6.0    parabolic block approaches its empirical blowup near alpha~7.
#   7.0         parabolic block catastrophic blowup.
#
# n_steps = n_steps_grad = 1000 matches the forward and gradient horizons
# for a fair precision claim.  1000 is long enough to reach the structural
# decay plateau (~N=100 on the parabolic block, ~N=700 on the linearized
# block) with clear margin, and short enough that the backward tape for
# fp64 at batch 1 fits comfortably in GPU memory.
python stability_probe.py \
    --n-steps 1000 \
    --n-steps-grad 1000 \
    --alphas 0.5 0.99 1.0 2.0 3.0 4.0 5.0 6.0 7.0 \
    --batch-size 4 \
    --seed 42

echo
echo "=== Compiling figure ==="
bash "$(dirname "$0")/process_results.sh"
