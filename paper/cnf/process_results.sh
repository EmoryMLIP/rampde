#!/bin/bash
#
# CNF Experiment Results Processing Script
#
# This script processes raw experimental data from the CNF experiment and generates:
# - Figure 2: CNF overview figure (wide format)
# - Table 2: CNF results table
#
# Usage:
#   ./process_results.sh [SEED]
#
# Arguments:
#   SEED: Random seed to filter results (default: 24 for test runs, use 42 for production)
#
# Outputs:
#   - outputs/fig_cnf_overview/cnf_overview_figure_wide.tex
#   - outputs/fig_cnf_overview/cnf_overview_figure_wide.pdf
#   - outputs/tab_cnf_results/cnf_results_table.tex
#   - outputs/tab_cnf_results/cnf_table_standalone.pdf

# Parse command line arguments
SEED=42  # Default to seed 24 (test runs)

echo "=================================="
echo "CNF Results Processing"
echo "=================================="
echo "Using seed: $SEED"
echo ""

# Check if we're in the right directory
if [ ! -f "cnf.py" ]; then
    echo "Error: Must run from paper/cnf/ directory"
    exit 1
fi

# Create output directories
mkdir -p outputs/fig_cnf_overview
mkdir -p outputs/tab_cnf_results

echo ""
echo "Step 0: Aggregating CNF experiment results..."
python aggregate_cnf_results.py --raw-data-dir raw_data/cnf --output raw_data/cnf/summary_cnf.csv --seed $SEED

if [ $? -eq 0 ]; then
    echo "✓ Results aggregated successfully"
else
    echo "✗ Failed to aggregate results"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 1: Generating CNF overview figure (Figure 2)..."
python generate_cnf_overview_wide.py

if [ $? -eq 0 ]; then
    echo "✓ CNF overview figure generated successfully"
else
    echo "✗ Failed to generate CNF overview figure"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 2: Extracting CNF subplot images..."
# Extract subplot images needed for the overview figure
output_dir="outputs/fig_cnf_overview"

# Configurations needed (precision_solver format)
declare -a configs=(
    "bfloat16_torchdiffeq"
    "bfloat16_rampde"
    "float16_none_torchdiffeq"
    "float16_none_rampde"
    "float16_grad_torchdiffeq"
    "float16_grad_rampde"
    "float16_dynamic_rampde"
)

# Datasets
declare -a datasets=("2spirals" "8gaussians" "checkerboard")

extracted_count=0
for dataset in "${datasets[@]}"; do
    for config in "${configs[@]}"; do
        # Find the latest experiment directory for this config with the specified seed
        pattern="raw_data/cnf/${dataset}_${config}_rk4_*_seed${SEED}_*"
        exp_dir=$(ls -dt $pattern 2>/dev/null | head -1)

        if [ -n "$exp_dir" ]; then
            # Find the final visualization image (highest iteration number)
            viz_img=$(ls -1 $exp_dir/cnf-viz-*.jpg 2>/dev/null | sort -V | tail -1)

            if [ -n "$viz_img" ]; then
                # Extract samples subplot
                python extract_cnf_subplots.py \
                    -i "$viz_img" \
                    -o "$output_dir" \
                    --prefix "${dataset}_${config}" \
                    --skip-target > /dev/null 2>&1

                if [ $? -eq 0 ]; then
                    ((extracted_count++))
                fi
            fi
        fi
    done
done

# Also extract target images (from tfloat32_torchdiffeq)
for dataset in "${datasets[@]}"; do
    pattern="raw_data/cnf/${dataset}_tfloat32_torchdiffeq_rk4_*_seed${SEED}_*"
    exp_dir=$(ls -dt $pattern 2>/dev/null | head -1)

    if [ -n "$exp_dir" ]; then
        viz_img=$(ls -1 $exp_dir/cnf-viz-*.jpg 2>/dev/null | sort -V | tail -1)

        if [ -n "$viz_img" ]; then
            python extract_cnf_subplots.py \
                -i "$viz_img" \
                -o "$output_dir" \
                --prefix "${dataset}_float32" \
                --target-samples > /dev/null 2>&1

            if [ $? -eq 0 ]; then
                ((extracted_count++))
            fi
        fi
    fi
done

echo "✓ Extracted $extracted_count subplot images"

echo ""
echo "Step 3: Generating CNF results table (Table 2)..."
python generate_experiment_table.py \
    --experiment-type cnf \
    --input raw_data/cnf/summary_cnf.csv

if [ $? -eq 0 ]; then
    echo "✓ CNF results table generated successfully"
else
    echo "✗ Failed to generate CNF results table"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 4: Compiling PDFs..."

# Compile figure PDF
cd outputs/fig_cnf_overview
pdflatex -interaction=batchmode cnf_overview_figure_wide.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ CNF overview figure PDF compiled"
else
    echo "✗ Failed to compile CNF overview figure PDF"
fi
cd ../..

# Compile table PDF
cd outputs/tab_cnf_results
pdflatex -interaction=batchmode cnf_table_standalone.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ CNF results table PDF compiled"
else
    echo "✗ Failed to compile CNF results table PDF"
fi
cd ../..

echo ""
echo "Step 5: Creating accessible CSV summary..."
# Copy the summary CSV to outputs for easy access
cp raw_data/cnf/summary_cnf.csv outputs/cnf_results_seed${SEED}.csv
if [ $? -eq 0 ]; then
    echo "✓ CSV summary copied to outputs/cnf_results_seed${SEED}.csv"
else
    echo "✗ Failed to copy CSV summary"
fi

echo ""
echo "=================================="
echo "CNF Processing Complete!"
echo "=================================="
echo ""
echo "Outputs saved to:"
echo "  - outputs/fig_cnf_overview/cnf_overview_figure_wide.pdf"
echo "  - outputs/tab_cnf_results/cnf_table_standalone.pdf"
echo "  - outputs/cnf_results_seed${SEED}.csv"
echo ""
echo "LaTeX sources:"
echo "  - outputs/fig_cnf_overview/cnf_overview_figure_wide.tex"
echo "  - outputs/tab_cnf_results/cnf_results_table.tex"

