#!/bin/bash
#
# Roundoff Error Analysis Results Processing Script
#
# This script processes raw experimental data from the roundoff error experiment and generates:
# - Figure 3: CNF roundoff error plots for various configurations
# - Combined 2x2 figure showing key comparisons
#
# Usage:
#   ./process_results.sh [--combined-only]
#
# Outputs:
#   - outputs/fig_cnf_roundoff/cnf_roundoff_*.tex
#   - outputs/fig_cnf_roundoff/cnf_roundoff_*.pdf
#   - outputs/fig_cnf_roundoff/cnf_roundoff_combined_2x2.pdf

set -e  # Exit on error

echo "=================================="
echo "Roundoff Error Analysis Processing"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "roundoff_cnf.py" ] && [ ! -f "plot_cnf_roundoff.py" ]; then
    echo "Error: Must run from paper/roundoff/ directory"
    exit 1
fi

# Create output directories
mkdir -p outputs/fig_cnf_roundoff

# Check if --combined-only flag is passed
if [[ "$*" != *"--combined-only"* ]]; then
    echo ""
    echo "Generating individual CNF roundoff plots for key configurations..."

    # Key configurations shown in the paper
    declare -a configs=(
        "rk4 float16 torchdiffeq grad"
        "rk4 float16 rampde dynamic"
        "euler float16 torchdiffeq none"
        "euler float16 torchdiffeq grad"
        "euler float16 rampde none"
        "euler float16 rampde grad"
        "euler float16 rampde dynamic"
        "euler bfloat16 torchdiffeq none"
        "euler bfloat16 rampde none"
    )

    for config in "${configs[@]}"; do
        read -r method precision odeint scaler <<< "$config"
        echo "  Processing: $method-$precision-$odeint-$scaler"
        python plot_cnf_roundoff.py \
            --method "$method" \
            --precision "$precision" \
            --odeint "$odeint" \
            --scaler "$scaler" || echo "  Warning: Failed for $config (non-critical)"
    done

    echo "✓ Individual roundoff plots generated"
else
    echo "Skipping individual plots (--combined-only flag)"
fi

echo ""
echo "Generating combined 2x2 CNF roundoff figure (Figure 3)..."
python plot_cnf_roundoff.py --create-combined

if [ $? -eq 0 ]; then
    echo "✓ Combined roundoff figure generated successfully"
else
    echo "✗ Failed to generate combined roundoff figure"
    exit 1
fi

echo ""
echo "=================================="
echo "Roundoff Analysis Complete!"
echo "=================================="
echo ""
echo "Outputs saved to:"
echo "  - outputs/fig_cnf_roundoff/"

