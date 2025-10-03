#!/bin/bash
#
# STL10 Experiment Results Processing Script
#
# This script processes raw experimental data from the STL10 experiment and generates:
# - Summary CSV: Aggregated metrics from all experiments
# - Training loss convergence plots (TikZ format)
# - Results summary table
#
# Usage:
#   ./process_results.sh [SEED]
#
# Arguments:
#   SEED: Random seed to filter results (default: 25 for production runs with 160 epochs)
#
# Outputs:
#   - raw_data/ode_stl10/summary_ode_stl10.csv
#   - outputs/fig_stl10_train_loss/stl10_train_loss_*.csv
#   - outputs/tab_stl10_results/stl10_results_table.tex
#   - outputs/tab_stl10_results/stl10_results_table_tabular.tex
#   - outputs/tab_stl10_results/stl10_results_table_standalone.tex
#   - outputs/tab_stl10_results/stl10_results_table_standalone.pdf

# Parse command line arguments
SEED=${1:-25}  # Default to seed 25 (160 epochs production runs)

echo "=================================="
echo "STL10 Results Processing"
echo "=================================="
echo "Using seed: $SEED"
echo ""

# Check if we're in the right directory
if [ ! -f "ode_stl10.py" ]; then
    echo "Error: Must run from paper/stl10/ directory"
    exit 1
fi

# Create output directories
mkdir -p outputs/fig_stl10_train_loss
mkdir -p outputs/tab_stl10_results

echo ""
echo "Step 1: Aggregating raw experiment data into summary CSV..."
python aggregate_stl10_results.py --raw-data-dir raw_data/ode_stl10 --output raw_data/ode_stl10/summary_ode_stl10.csv --seed $SEED

if [ $? -eq 0 ]; then
    echo "✓ Results aggregated successfully"
else
    echo "✗ Failed to aggregate results"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 2: Generating STL10 convergence plots..."
python plot_stl10_convergence.py --filter "*seed${SEED}*"

if [ $? -eq 0 ]; then
    echo "✓ STL10 convergence plots generated successfully"
else
    echo "✗ Failed to generate STL10 convergence plots"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 3: Generating STL10 results table..."
python generate_stl10_table.py \
    --csv-file ./raw_data/ode_stl10/summary_ode_stl10.csv \
    --results-dir ./raw_data/ode_stl10 \
    --width 128 \
    --seed $SEED

if [ $? -eq 0 ]; then
    echo "✓ STL10 results table generated successfully"
else
    echo "✗ Failed to generate STL10 results table"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 4: Compiling PDFs..."

# Compile convergence plot (needs 2 runs for \ref resolution)
cd outputs/fig_stl10_train_loss
if [ -f "stl10_train_loss_standalone.tex" ]; then
    pdflatex -interaction=batchmode stl10_train_loss_standalone.tex > /dev/null 2>&1
    pdflatex -interaction=batchmode stl10_train_loss_standalone.tex > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ STL10 convergence plot PDF compiled"
    else
        echo "✗ Failed to compile STL10 convergence plot PDF"
    fi
else
    echo "  Skipping convergence plot (file not found)"
fi
cd ../..

# Compile table
cd outputs/tab_stl10_results
pdflatex -interaction=batchmode stl10_results_table_standalone.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ STL10 results table PDF compiled"
else
    echo "✗ Failed to compile STL10 results table PDF"
fi
cd ../..

echo ""
echo "Step 5: Creating accessible CSV summary..."
cp raw_data/ode_stl10/summary_ode_stl10.csv outputs/stl10_results_seed${SEED}.csv
if [ $? -eq 0 ]; then
    echo "✓ CSV summary copied to outputs/stl10_results_seed${SEED}.csv"
else
    echo "✗ Failed to copy CSV summary"
fi

echo ""
echo "=================================="
echo "STL10 Processing Complete!"
echo "=================================="
echo ""
echo "Outputs saved to:"
echo "  - outputs/fig_stl10_train_loss/stl10_train_loss_standalone.pdf"
echo "  - outputs/tab_stl10_results/stl10_results_table_standalone.pdf"
echo "  - outputs/stl10_results_seed${SEED}.csv"
echo ""
echo "LaTeX sources:"
echo "  - outputs/fig_stl10_train_loss/stl10_train_loss_standalone.tex"
echo "  - outputs/tab_stl10_results/stl10_results_table.tex"
echo "  - outputs/tab_stl10_results/stl10_results_table_tabular.tex"
echo "  - outputs/tab_stl10_results/stl10_results_table_standalone.tex"

