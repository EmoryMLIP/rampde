#!/usr/bin/env python3
"""
Generate LaTeX table from experiment results.

This script processes experiment summary data and creates a professional LaTeX table
showing performance metrics across different numerical configurations.
Supports multiple experiment types: CNF, OTFlowLarge, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def format_number(value, precision=3, scientific_threshold=1e-3):
    """Format number with appropriate precision and scientific notation."""
    if pd.isna(value):
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


def get_experiment_metrics(experiment_type):
    """Get metrics and their formatting for different experiment types."""
    if experiment_type == 'cnf':
        return [
            ('val_loss', 'Val Loss', lambda x: format_number(x, 3)),
            ('time_fwd', 'Avg Fwd Time (s)', lambda x: format_number(x, 2)),
            ('time_bwd', 'Avg Bwd Time (s)', lambda x: format_number(x, 2)), 
            ('total_time', 'Total Time (s)', lambda x: format_number(x, 1)),
            ('max_memory_mb', 'Max Memory', lambda x: format_memory(x))
        ]
    elif experiment_type == 'otflowlarge':
        return [
            ('val_loss', 'Val Loss', lambda x: format_number(x, 1)),
            ('val_L', 'Val L', lambda x: format_number(x, 3)),
            ('val_NLL', 'Val NLL', lambda x: format_number(x, 1)),
            ('val_HJB', 'Val HJB', lambda x: format_number(x, 3)),
            ('time_fwd', 'Avg Fwd Time (s)', lambda x: format_number(x, 3)),
            ('time_bwd', 'Avg Bwd Time (s)', lambda x: format_number(x, 3)), 
            ('total_time', 'Total Time (s)', lambda x: format_number(x, 1)),
            ('max_memory_mb', 'Max Memory', lambda x: format_memory(x)),
            ('val_mmd', 'Val MMD', lambda x: format_number(x, 4))
        ]
    else:
        # Default metrics
        return [
            ('val_loss', 'Val Loss', lambda x: format_number(x, 3)),
            ('time_fwd', 'Avg Fwd Time (s)', lambda x: format_number(x, 2)),
            ('time_bwd', 'Avg Bwd Time (s)', lambda x: format_number(x, 2)),
            ('max_memory_mb', 'Max Memory', lambda x: format_memory(x))
        ]


def get_experiment_info(experiment_type):
    """Get experiment-specific information for table generation."""
    if experiment_type == 'cnf':
        return {
            'title': 'Continuous Normalizing Flow (CNF) experiment results',
            'description': 'across different datasets and numerical precision configurations.',
            'label': 'tab:cnf_results'
        }
    elif experiment_type == 'otflowlarge':
        return {
            'title': 'OT-Flow experiments on large-scale datasets',
            'description': 'with different numerical precision configurations.',
            'label': 'tab:otflowlarge_results'
        }
    else:
        return {
            'title': f'{experiment_type.upper()} experiment results',
            'description': 'across different numerical precision configurations.',
            'label': f'tab:{experiment_type}_results'
        }


def load_and_process_data(csv_file, experiment_type):
    """Load experiment summary data and process it for table generation."""
    df = pd.read_csv(csv_file)
    
    # Add configuration column
    df['config'] = df.apply(parse_precision_config, axis=1)
    
    # Calculate total time if needed for CNF (otflowlarge already has it)
    if experiment_type == 'cnf' and 'total_time' not in df.columns:
        df['total_time'] = df['time_fwd_sum'] + df['time_bwd_sum']
    
    # Get available columns for this experiment type
    metrics = get_experiment_metrics(experiment_type)
    required_cols = ['data_name', 'config'] + [metric[0] for metric in metrics]
    available_cols = [col for col in required_cols if col in df.columns]
    
    results = df[available_cols].copy()
    return results


def create_latex_table(data, output_file, experiment_type):
    """Create LaTeX table from processed data with multi-level headers."""
    
    # Filter out FP32 configurations if desired
    if experiment_type == 'cnf':
        data_filtered = data[~data['config'].str.contains('TFP32')].copy()
    else:
        data_filtered = data[~data['config'].str.contains('FP32')].copy()

    # Get unique datasets
    datasets = sorted(data_filtered['data_name'].unique())
    
    # Define the column structure (same across all experiment types)
    if experiment_type == 'cnf':
        column_structure = [
            # FP32 columns
            ('FP32 (torchdiffeq)', 'FP32', 'torchdiffeq', ''),
            ('FP32 (rampde)', 'FP32', 'rampde', ''),
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
    else:
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
    
    # Get experiment-specific info
    exp_info = get_experiment_info(experiment_type)
    metrics = get_experiment_metrics(experiment_type)
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append(f"% {experiment_type.upper()} Experiment Results")
    latex_lines.append(f"% Generated automatically from summary_{experiment_type}.csv")
    latex_lines.append("")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{")
    latex_lines.append(f"{exp_info['title']} with mixed-precision neural ODEs. ")
    latex_lines.append(f"{exp_info['description']} ")
    latex_lines.append("Dynamic scaling (Dyn) enables robust float16 training, ")
    latex_lines.append("while gradient scaling (Grad) provides an alternative when dynamic scaling is unavailable.")
    latex_lines.append("}")
    latex_lines.append(f"\\label{{{exp_info['label']}}}")
    latex_lines.append("")
    
    # Table structure: 1 column for metrics + 9 data columns
    col_spec = "l" + "c" * 9
    
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Multi-level headers
    # Level 1: Precision types - use full precision names
    if experiment_type == 'cnf':
        header1 = "& \\multicolumn{2}{c}{float32} & \\multicolumn{2}{c}{bfloat16} & \\multicolumn{5}{c}{float16} \\\\"
    else:
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
        
        # Dataset name row
        dataset_name = dataset.replace('_', '\\_')
        dataset_row = f"\\textbf{{{dataset_name}}} & & & & & & & & & \\\\"
        latex_lines.append(dataset_row)
        latex_lines.append("\\addlinespace[0.5ex]")
        
        # Filter metrics to only include those with available data
        available_metrics = []
        for metric_col, metric_name, formatter in metrics:
            if metric_col in data.columns:
                available_metrics.append((metric_col, metric_name, formatter))
        
        for metric_col, metric_name, formatter in available_metrics:
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


def create_standalone_test(table_file, standalone_file, experiment_type):
    """Create standalone LaTeX file for testing."""
    
    latex_content = f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{graphicx}}
\\usepackage[margin=1in]{{geometry}}

\\begin{{document}}

\\title{{{experiment_type.upper()} Results Table Test}}
\\author{{Generated}}
\\date{{\\today}}
\\maketitle

\\input{{{table_file.stem}}}

\\end{{document}}
"""
    
    with open(standalone_file, 'w') as f:
        f.write(latex_content)
    
    print(f"Standalone test file written to: {standalone_file}")
    return standalone_file


def main():
    parser = argparse.ArgumentParser(description='Generate experiment results LaTeX table')
    parser.add_argument('--input', required=True,
                       help='Input CSV file with experiment results')
    parser.add_argument('--output', 
                       help='Output LaTeX table file (default: auto-generated)')
    parser.add_argument('--standalone', 
                       help='Standalone test LaTeX file (default: auto-generated)')
    parser.add_argument('--experiment-type', type=str, default='cnf',
                       choices=['cnf', 'otflowlarge', 'ode_stl10'],
                       help='Type of experiment')
    args = parser.parse_args()
    
    # Process paths
    input_file = Path(args.input)
    
    if args.output:
        output_file = Path(args.output)
    else:
        # Output to experiment-local directory
        output_dir = Path(f"outputs/tab_{args.experiment_type}_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{args.experiment_type}_results_table.tex"

    if args.standalone:
        standalone_file = Path(args.standalone)
    else:
        # Use same directory as main output
        output_dir = output_file.parent
        standalone_file = output_dir / f"{args.experiment_type}_table_standalone.tex"
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return 1
    
    # Load and process data
    print(f"Loading data from: {input_file}")
    data = load_and_process_data(input_file, args.experiment_type)
    print(f"Processed {len(data)} experiment results")
    print(f"Datasets: {sorted(data['data_name'].unique())}")
    print(f"Configurations: {len(data['config'].unique())} unique")
    print(data['config'].unique())
    # Generate LaTeX table
    create_latex_table(data, output_file, args.experiment_type)
    
    # Create standalone test file
    create_standalone_test(output_file, standalone_file, args.experiment_type)
    
    print("\nFiles generated successfully!")
    print(f"1. LaTeX table: {output_file}")
    print(f"2. Standalone test: {standalone_file}")
    print(f"\nTo test compilation: pdflatex {standalone_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())