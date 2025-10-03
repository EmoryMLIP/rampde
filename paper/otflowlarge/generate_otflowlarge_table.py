#!/usr/bin/env python3
"""
Generate LaTeX table from OTFlowLarge experiment results.

This script processes the OTFlowLarge summary data and creates a professional LaTeX table
showing performance metrics across different numerical configurations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def format_number(value, precision=3, scientific_threshold=1e-3):
    """Format number with appropriate precision and scientific notation."""
    if pd.isna(value) or value == 'Failed':
        return "---"
    
    if abs(value) < scientific_threshold and value != 0:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"


def format_time(seconds, precision=2):
    """Format time in seconds with appropriate unit."""
    if pd.isna(seconds):
        return "---"
    return f"{seconds:.{precision}f}s"


def format_memory(mb, precision=1):
    """Format memory in MB."""
    if pd.isna(mb):
        return "---"
    if mb >= 1000:
        return f"{mb/1000:.{precision}f}GB"
    else:
        return f"{mb:.{precision}f}MB"


def parse_precision_config(row):
    """Parse precision configuration from data row."""
    precision = row['precision_str']
    scaler = row['scaler_name'] if pd.notna(row['scaler_name']) else ''
    odeint = row['odeint_type']
    
    # Create readable configuration names
    if precision == 'float32':
        return f"FP32 ({odeint})"
    elif precision == 'tfloat32':
        return f"TF32 ({odeint})"
    elif precision == 'bfloat16':
        return f"BF16 ({odeint})"
    elif precision == 'float16':
        if scaler == 'dynamic':
            return f"FP16-Dyn ({odeint})"
        elif scaler == 'grad':
            return f"FP16-Grad ({odeint})"
        elif scaler == 'none' or scaler == '':
            return f"FP16-None ({odeint})"
        else:
            return f"FP16-{scaler} ({odeint})"
    else:
        return f"{precision} ({odeint})"


def load_and_process_data(csv_file):
    """Load OTFlowLarge summary data and process it for table generation."""
    df = pd.read_csv(csv_file)
    
    # Add configuration column
    df['config'] = df.apply(parse_precision_config, axis=1)
    
    # Calculate total time
    df['total_time'] = df['time_fwd_sum'] + df['time_bwd_sum']
    
    # Calculate average time per step (sum of fwd and bwd)
    df['avg_time_per_step'] = df['time_fwd'] + df['time_bwd']
    
    # Select required columns for OTFlowLarge
    results = df[['data_name', 'config', 'val_loss', 'val_L', 'val_NLL', 'val_HJB', 
                  'time_fwd', 'time_bwd', 'avg_time_per_step', 'total_time', 'max_memory_mb', 
                  'iter', 'val_mmd']].copy()
    
    # Add missing/failed configurations
    # Check if we have the expected configurations and add failed ones if missing
    expected_configs = [
        ('FP16-None (torchdiffeq)', 'float16', 'torchdiffeq', 'none'),
        # Add other expected configs if needed
    ]
    
    for config_name, precision, solver, scaler in expected_configs:
        if config_name not in results['config'].values:
            # Add failed experiment entry
            failed_row = {
                'data_name': 'bsds300',  # OTFlowLarge only has bsds300
                'config': config_name,
                'val_loss': 'Failed',
                'val_L': 'Failed', 
                'val_NLL': 'Failed',
                'val_HJB': 'Failed',
                'time_fwd': 'Failed',
                'time_bwd': 'Failed', 
                'avg_time_per_step': 'Failed',
                'total_time': 'Failed',
                'max_memory_mb': 'Failed',
                'iter': 'Failed',
                'val_mmd': 'Failed'
            }
            results = pd.concat([results, pd.DataFrame([failed_row])], ignore_index=True)
    
    return results


def create_latex_table(data, output_file):
    """Create LaTeX table from processed data with multi-level headers."""
    
    # Filter out FP32 configurations if desired
    data_filtered = data[~data['config'].str.contains('FP32')].copy()
    
    # Get unique datasets
    datasets = sorted(data_filtered['data_name'].unique())
    
    # Define the column structure (same as CNF but reordered for clarity)
    column_structure = [
        # TF32 columns
        ('TF32 (torchdiffeq)', 'TF32', 'torchdiffeq', ''),
        ('TF32 (rampde)', 'TF32', 'rampde', ''),
        # BF16 columns  
        ('BF16 (torchdiffeq)', 'BF16', 'torchdiffeq', ''),
        ('BF16 (rampde)', 'BF16', 'rampde', ''),
        # FP16 columns - grouped by solver
        ('FP16-Grad (torchdiffeq)', 'FP16', 'torchdiffeq', 'Grad'),
        ('FP16-None (torchdiffeq)', 'FP16', 'torchdiffeq', 'None'),
        ('FP16-Grad (rampde)', 'FP16', 'rampde', 'Grad'),
        ('FP16-None (rampde)', 'FP16', 'rampde', 'None'),
        ('FP16-Dyn (rampde)', 'FP16', 'rampde', 'Dyn'),
    ]
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append("% OTFlowLarge Experiment Results")
    latex_lines.append("% Generated automatically from summary_otflowlarge.csv")
    latex_lines.append("")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{")
    latex_lines.append("Large-scale optimal transport flow performance with mixed-precision neural ODEs. ")
    latex_lines.append("Our rampde implementation demonstrates significant memory efficiency gains ")
    latex_lines.append("while maintaining competitive validation loss across different precision formats. ")
    latex_lines.append("Dynamic scaling (Dyn) enables robust float16 training with minimal accuracy degradation, ")
    latex_lines.append("while gradient scaling (Grad) provides a viable alternative for challenging datasets.")
    latex_lines.append("}")
    latex_lines.append("\\label{tab:otflowlarge_results}")
    latex_lines.append("")
    
    # Table structure: 1 column for metrics + 9 data columns
    col_spec = "l" + "c" * 9
    
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Multi-level headers
    # Level 1: Precision types - use full precision names
    header1 = "& \\multicolumn{2}{c}{tfloat32} & \\multicolumn{2}{c}{bfloat16} & \\multicolumn{5}{c}{float16} \\\\"
    latex_lines.append(header1)
    
    # Level 2: Solver types
    header2 = "& \\multicolumn{1}{c}{torchdiffeq} & \\multicolumn{1}{c}{rampde} & \\multicolumn{1}{c}{torchdiffeq} & \\multicolumn{1}{c}{rampde} & \\multicolumn{2}{c}{torchdiffeq} & \\multicolumn{3}{c}{rampde} \\\\"
    latex_lines.append(header2)
    
    # Level 3: Scaler types (only for FP16)
    header3 = "Metric & & & & & Grad & None & Grad & None & Dyn \\\\"
    latex_lines.append(header3)
    latex_lines.append("\\midrule")
    
    # Data rows for each dataset
    for dataset_idx, dataset in enumerate(datasets):
        dataset_data = data_filtered[data_filtered['data_name'] == dataset]
        
        # Add blank row and dataset name
        if dataset_idx > 0:
            latex_lines.append("\\addlinespace")
        
        # Skip dataset name row - will be explained in caption
        
        # Metrics specific to OTFlowLarge
        metrics = [
            ('val_loss', 'Val Loss', lambda x: format_number(x, 1)),
            ('val_L', 'Val L', lambda x: format_number(x, 3)),
            ('val_NLL', 'Val NLL', lambda x: format_number(x, 1)),
            ('val_HJB', 'Val HJB', lambda x: format_number(x, 3)),
            ('val_mmd', 'Val MMD', lambda x: format_number(x, 4)),
            ('time_fwd', 'Avg Fwd Time (s)', lambda x: format_number(x, 3)),
            ('time_bwd', 'Avg Bwd Time (s)', lambda x: format_number(x, 3)), 
            ('avg_time_per_step', 'Avg Time per Step (s)', lambda x: format_number(x, 3)),
            ('total_time', 'Total Time (s)', lambda x: format_number(x, 1)),
            ('iter', 'Total Iterations', lambda x: f"{int(x):,}" if pd.notna(x) and str(x) != 'Failed' else "---"),
            ('max_memory_mb', 'Max Memory (GB)', lambda x: "---" if str(x) == 'Failed' else f"{x/1000:.1f}" if pd.notna(x) and x >= 1000 else f"{x:.0f}MB" if pd.notna(x) else "---")
        ]
        
        for metric_col, metric_name, formatter in metrics:
            row = f"\\quad {metric_name}"
            
            # Add data for each configuration in the specified order
            for config_name, precision, solver, scaler in column_structure:
                config_data = dataset_data[dataset_data['config'] == config_name]
                if not config_data.empty:
                    value = config_data[metric_col].iloc[0]
                    formatted_value = formatter(value)
                else:
                    formatted_value = "---"
                row += f" & {formatted_value}"
            
            row += " \\\\"
            latex_lines.append(row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}%")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table written to: {output_file}")
    return output_file


def create_standalone_test(table_file, standalone_file):
    """Create standalone LaTeX file for testing."""
    
    latex_content = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{graphicx}
\\usepackage[margin=1in]{geometry}

\\begin{document}

\\title{OTFlowLarge Results Table Test}
\\author{Generated}
\\date{\\today}
\\maketitle

\\input{otflowlarge_results_table}

\\end{document}
"""
    
    with open(standalone_file, 'w') as f:
        f.write(latex_content)
    
    print(f"Standalone test file written to: {standalone_file}")
    return standalone_file


def main():
    parser = argparse.ArgumentParser(description='Generate OTFlowLarge results LaTeX table')
    parser.add_argument('--input', default='./raw_data/otflowlarge/summary_otflowlarge.csv',
                       help='Input CSV file with OTFlowLarge results')
    parser.add_argument('--output', default='./outputs/tab_otflowlarge_results/otflowlarge_results_table.tex',
                       help='Output LaTeX table file')
    parser.add_argument('--standalone', default='./outputs/tab_otflowlarge_results/otflowlarge_table_standalone.tex',
                       help='Standalone test LaTeX file')
    args = parser.parse_args()
    
    # Process paths
    input_file = Path(args.input)
    output_file = Path(args.output)
    standalone_file = Path(args.standalone)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return 1
    
    # Load and process data
    print(f"Loading data from: {input_file}")
    data = load_and_process_data(input_file)
    print(f"Processed {len(data)} experiment results")
    print(f"Datasets: {sorted(data['data_name'].unique())}")
    print(f"Configurations: {len(data['config'].unique())} unique")
    
    # Generate LaTeX table
    create_latex_table(data, output_file)
    
    # Create standalone test file
    create_standalone_test(output_file, standalone_file)
    
    print("\nFiles generated successfully!")
    print(f"1. LaTeX table: {output_file}")
    print(f"2. Standalone test: {standalone_file}")
    print(f"\nTo test compilation: pdflatex {standalone_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())