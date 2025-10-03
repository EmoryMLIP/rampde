#!/bin/bash
#
# OTFlowLarge Experiment Results Processing Script
#
# This script processes raw experimental data from the OTFlowLarge experiment and generates:
# - Table 3: OTFlowLarge results table
#
# Usage:
#   ./process_results.sh [SEED]
#
# Arguments:
#   SEED: Random seed to filter results (default: 23 for test runs, use production seed as needed)
#
# Outputs:
#   - outputs/tab_otflowlarge_results/otflowlarge_results_table.tex
#   - outputs/tab_otflowlarge_results/otflowlarge_table_standalone.pdf
#   - outputs/otflowlarge_results_seed{N}.csv

# Parse command line arguments
SEED=${1:-23}  # Default to seed 23

echo "=================================="
echo "OTFlowLarge Results Processing"
echo "=================================="
echo "Using seed: $SEED"
echo ""

# Check if we're in the right directory
if [ ! -f "otflowlarge.py" ]; then
    echo "Error: Must run from paper/otflowlarge/ directory"
    exit 1
fi

# Create output directories
mkdir -p outputs/tab_otflowlarge_results

echo ""
echo "Step 1: Filtering results by seed..."
# Filter the summary CSV by seed
input_csv="raw_data/otflowlarge/summary_otflowlarge.csv"
filtered_csv="raw_data/otflowlarge/summary_otflowlarge_seed${SEED}.csv"

if [ ! -f "$input_csv" ]; then
    echo "✗ Error: Summary CSV not found: $input_csv"
    exit 1
fi

# Filter by seed using grep (keep header + matching rows)
head -1 "$input_csv" > "$filtered_csv"
grep "_seed${SEED}_" "$input_csv" >> "$filtered_csv" 2>/dev/null || true

# Count filtered results
result_count=$(wc -l < "$filtered_csv")
result_count=$((result_count - 1))  # Subtract header

if [ $result_count -eq 0 ]; then
    echo "✗ Warning: No experiments found for seed ${SEED}"
else
    echo "✓ Filtered $result_count experiments for seed ${SEED}"
fi

echo ""
echo "Step 2: Generating OTFlowLarge results table (Table 3)..."
python generate_otflowlarge_table.py --input "$filtered_csv"

if [ $? -eq 0 ]; then
    echo "✓ OTFlowLarge results table generated successfully"
else
    echo "✗ Failed to generate OTFlowLarge results table"
    echo "  Continuing anyway..."
fi

echo ""
echo "Step 3: Compiling PDF..."
cd outputs/tab_otflowlarge_results
pdflatex -interaction=batchmode otflowlarge_table_standalone.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ OTFlowLarge results table PDF compiled"
else
    echo "✗ Failed to compile OTFlowLarge results table PDF"
fi
cd ../..

echo ""
echo "Step 4: Creating accessible CSV summary..."
# Copy the filtered summary CSV to outputs for easy access
cp "$filtered_csv" outputs/otflowlarge_results_seed${SEED}.csv
if [ $? -eq 0 ]; then
    echo "✓ CSV summary copied to outputs/otflowlarge_results_seed${SEED}.csv"
else
    echo "✗ Failed to copy CSV summary"
fi

echo ""
echo "=================================="
echo "OTFlowLarge Processing Complete!"
echo "=================================="
echo ""
echo "Outputs saved to:"
echo "  - outputs/tab_otflowlarge_results/otflowlarge_table_standalone.pdf"
echo "  - outputs/otflowlarge_results_seed${SEED}.csv"
echo ""
echo "LaTeX source:"
echo "  - outputs/tab_otflowlarge_results/otflowlarge_results_table.tex"

