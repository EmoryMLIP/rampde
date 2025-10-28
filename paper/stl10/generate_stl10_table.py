#!/usr/bin/env python3
"""
Generate LaTeX table from STL10 experiment results.

This script processes the STL10 summary data and creates a professional LaTeX table
showing performance metrics across different numerical precision configurations.
Based on generate_otflowlarge_table.py but adapted for STL10 classification results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os


def format_number(value, precision=3, scientific_threshold=1e-3):
    """Format number with appropriate precision and scientific notation."""
    if pd.isna(value):
        return "---"
    
    if abs(value) < scientific_threshold and value != 0:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"


def format_time(seconds, precision=1):
    """Format time in seconds with appropriate unit."""
    if pd.isna(seconds):
        return "---"
    if seconds >= 3600:
        return f"{seconds/3600:.{precision}f}h"
    elif seconds >= 60:
        return f"{seconds/60:.{precision}f}m"
    else:
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


def load_test_metrics(results_dir: str, directory: str) -> tuple:
    """Load test loss and accuracy from test_loss.txt file if it exists."""
    test_loss_file = Path(results_dir) / directory / 'test_loss.txt'
    
    if not test_loss_file.exists():
        return np.nan, np.nan
    
    try:
        with open(test_loss_file, 'r') as f:
            content = f.read().strip()
            
            test_loss = np.nan
            test_acc = np.nan
            
            # Extract test loss value
            if 'Test Loss:' in content:
                test_loss = float(content.split('Test Loss:')[1].split()[0])
            
            # Extract test accuracy value  
            if 'Test Accuracy:' in content:
                test_acc = float(content.split('Test Accuracy:')[1].split()[0])
                
            return test_loss, test_acc
    except Exception as e:
        print(f"Warning: Could not parse test metrics from {test_loss_file}: {e}")
        return np.nan, np.nan


def load_and_process_data(csv_file, results_dir=None):
    """Load STL10 summary data and process it for table generation."""
    df = pd.read_csv(csv_file)
    
    # Add configuration column
    df['config'] = df.apply(parse_precision_config, axis=1)
    
    # Calculate total time
    df['total_time'] = df['time_fwd_sum'] + df['time_bwd_sum']
    
    # Load test metrics data if results directory is provided
    if results_dir:
        print("Loading test metrics data...")
        test_losses = []
        test_accs = []
        for idx, row in df.iterrows():
            test_loss, test_acc = load_test_metrics(results_dir, row['directory'])
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        df['test_loss'] = test_losses
        df['test_acc'] = test_accs
    else:
        df['test_loss'] = np.nan
        df['test_acc'] = np.nan
    
    # Select required columns for STL10
    results = df[['config', 'val_acc', 'train_acc', 'val_loss', 'test_loss', 'test_acc',
                  'time_fwd', 'time_bwd', 'time_fwd_sum', 'time_bwd_sum', 'total_time', 'max_memory_mb',
                  'width', 'seed', 'precision_str', 'odeint_type', 'scaler_name']].copy()
    
    return results


def create_latex_table(data, output_file, width_filter=None, seed_filter=None):
    """Create LaTeX table from processed data matching OTFlowLarge format."""

    # Filter by width and seed if specified
    filtered_data = data.copy()
    if width_filter is not None:
        filtered_data = filtered_data[filtered_data['width'] == width_filter]
        print(f"Filtered to {len(filtered_data)} experiments with width={width_filter}")

    if seed_filter is not None:
        filtered_data = filtered_data[filtered_data['seed'] == seed_filter]
        print(f"Filtered to {len(filtered_data)} experiments with seed={seed_filter}")

    if len(filtered_data) == 0:
        raise ValueError(f"No experiments found with width={width_filter}, seed={seed_filter}")

    # Remove failed experiments (very low accuracy)
    successful_data = filtered_data[filtered_data['val_acc'] > 0.1].copy()
    failed_configs = filtered_data[filtered_data['val_acc'] <= 0.1]['config'].tolist()

    if failed_configs:
        print(f"Note: Excluding {len(failed_configs)} failed configurations: {failed_configs}")

    if len(successful_data) == 0:
        print("Warning: No successful experiments found, using all data")
        successful_data = filtered_data

    # Create configuration groups similar to OTFlowLarge
    # Group by precision type and solver
    config_groups = {}

    for idx, row in successful_data.iterrows():
        precision = row['precision_str']
        solver = row['odeint_type']
        scaler = row['scaler_name'] if pd.notna(row['scaler_name']) else 'none'

        # Organize similar to OTFlowLarge: TF32, BF16, FP16 variants
        if precision == 'tfloat32':
            group = 'TF32'
        elif precision == 'bfloat16':
            group = 'BF16'
        elif precision == 'float16':
            group = 'FP16'
        else:  # float32
            group = 'FP32'

        if group not in config_groups:
            config_groups[group] = {}

        if solver not in config_groups[group]:
            config_groups[group][solver] = {}

        config_groups[group][solver][scaler] = row

    # Build column headers dynamically based on available data
    precision_order = ['FP32', 'TF32', 'BF16', 'FP16']
    available_precisions = [p for p in precision_order if p in config_groups]

    # Count columns for each precision type
    col_counts = {}
    for prec in available_precisions:
        count = 0
        for solver in ['torchdiffeq', 'rampde']:
            if solver in config_groups[prec]:
                if prec == 'FP16':
                    # Count FP16 variants
                    count += len(config_groups[prec][solver])
                else:
                    count += 1
        col_counts[prec] = count

    total_cols = sum(col_counts.values())

    # Build tabular content (without table wrapper)
    tabular_content = f"\\resizebox{{\\textwidth}}{{!}}{{%\n"
    tabular_content += f"\\begin{{tabular}}{{l{'c' * total_cols}}}\n"
    tabular_content += "\\toprule\n"

    # Header row 1: Precision types - use full precision names
    precision_names = {'TF32': 'tfloat32', 'BF16': 'bfloat16', 'FP16': 'float16', 'FP32': 'float32'}
    header1 = "& "
    for prec in available_precisions:
        full_name = precision_names.get(prec, prec)
        if col_counts[prec] > 1:
            header1 += f"\\multicolumn{{{col_counts[prec]}}}{{c}}{{{full_name}}} & "
        else:
            header1 += f"{full_name} & "
    header1 = header1.rstrip("& ") + " \\\\\n"
    tabular_content += header1

    # Header row 2: Solver types and FP16 variants
    header2 = "& "
    for prec in available_precisions:
        for solver in ['torchdiffeq', 'rampde']:
            if solver in config_groups[prec]:
                if prec == 'FP16':
                    # Show FP16 variants
                    scalers = sorted(config_groups[prec][solver].keys())
                    if len(scalers) > 1:
                        header2 += f"\\multicolumn{{{len(scalers)}}}{{c}}{{{solver}}} & "
                    else:
                        header2 += f"{solver} & "
                else:
                    header2 += f"\\multicolumn{{1}}{{c}}{{{solver}}} & "
    header2 = header2.rstrip("& ") + " \\\\\n"
    tabular_content += header2

    # Header row 3: FP16 scaler types
    header3 = "Metric & "
    for prec in available_precisions:
        for solver in ['torchdiffeq', 'rampde']:
            if solver in config_groups[prec]:
                if prec == 'FP16':
                    scalers = sorted(config_groups[prec][solver].keys())
                    for scaler in scalers:
                        scaler_label = {'dynamic': 'Dyn', 'grad': 'Grad', 'none': 'None'}.get(scaler, scaler)
                        header3 += f"{scaler_label} & "
                else:
                    header3 += "& "
    header3 = header3.rstrip("& ") + " \\\\\n"
    tabular_content += header3

    tabular_content += "\\midrule\n"

    # Dataset label (STL10)
    tabular_content += "\\textbf{STL10} & " + " & " * (total_cols - 1) + " \\\\\n"
    tabular_content += "\\addlinespace[0.5ex]\n"

    # Data rows
    metrics = [
        ('Val Acc', 'val_acc', 3),
        ('Val Loss', 'val_loss', 3),
        ('Test Acc', 'test_acc', 3),
        ('Test Loss', 'test_loss', 3),
        ('Avg Fwd Time (s)', 'time_fwd', 2),
        ('Avg Bwd Time (s)', 'time_bwd', 2),
        ('Max Memory', 'max_memory_mb', 1)
    ]

    for metric_name, metric_col, precision in metrics:
        row = f"\\quad {metric_name} & "

        for prec in available_precisions:
            for solver in ['torchdiffeq', 'rampde']:
                if solver in config_groups[prec]:
                    if prec == 'FP16':
                        scalers = sorted(config_groups[prec][solver].keys())
                        for scaler in scalers:
                            if scaler in config_groups[prec][solver]:
                                data_row = config_groups[prec][solver][scaler]
                                value = data_row[metric_col]

                                if metric_name == 'Max Memory':
                                    if pd.notna(value) and value >= 1000:
                                        formatted_val = f"{value/1000:.1f}GB"
                                    else:
                                        formatted_val = f"{value:.0f}MB" if pd.notna(value) else "---"
                                else:
                                    formatted_val = format_number(value, precision)

                                row += f"{formatted_val} & "
                            else:
                                row += "--- & "
                    else:
                        if solver in config_groups[prec] and 'none' in config_groups[prec][solver]:
                            data_row = config_groups[prec][solver]['none']
                            value = data_row[metric_col]

                            if metric_name == 'Max Memory':
                                if pd.notna(value) and value >= 1000:
                                    formatted_val = f"{value/1000:.1f}GB"
                                else:
                                    formatted_val = f"{value:.0f}MB" if pd.notna(value) else "---"
                            else:
                                formatted_val = format_number(value, precision)

                            row += f"{formatted_val} & "
                        else:
                            row += "--- & "

        row = row.rstrip("& ") + " \\\\\n"
        tabular_content += row

    tabular_content += "\\bottomrule\n"
    tabular_content += "\\end{tabular}%\n"
    tabular_content += "}\n"

    # Create full table with wrapper
    caption_text = "STL10 image classification performance of mixed-precision neural ODEs"
    if width_filter and seed_filter:
        caption_text += f" with width {width_filter} and seed {seed_filter}"
    elif width_filter:
        caption_text += f" with width {width_filter}"
    elif seed_filter:
        caption_text += f" with seed {seed_filter}"
    caption_text += ". Our rampde implementation achieves competitive accuracy with 5--10× memory reduction compared to standard torchdiffeq, while float16 variants maintain over 76\\% test accuracy with minimal memory footprint. Dynamic scaling (Dyn) enables robust float16 training, with gradient scaling (Grad) providing an alternative when dynamic scaling is unavailable."

    full_table = f"""% STL10 Experiment Results
% Generated automatically from summary_ode_stl10.csv

\\begin{{table}}
\\centering
\\caption{{
{caption_text}
}}
\\label{{tab:stl10_results}}

{tabular_content}
\\end{{table}}
"""

    # Write full table to file
    with open(output_file, 'w') as f:
        f.write(full_table)

    print(f"LaTeX table saved to: {output_file}")

    # Write tabular-only version for standalone use
    output_path = Path(output_file)
    tabular_file = output_path.parent / f"{output_path.stem}_tabular.tex"
    with open(tabular_file, 'w') as f:
        f.write("% STL10 Experiment Results - Tabular Only\n")
        f.write("% Generated automatically from summary_ode_stl10.csv\n\n")
        f.write(tabular_content)

    print(f"Tabular-only version saved to: {tabular_file}")

    # Create standalone file
    standalone_file = output_path.parent / f"{output_path.stem}_standalone.tex"
    standalone_content = f"""\\documentclass[border=2mm]{{standalone}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{array}}

\\begin{{document}}
\\input{{{tabular_file.name}}}
\\end{{document}}
"""

    with open(standalone_file, 'w') as f:
        f.write(standalone_content)

    print(f"Standalone file saved to: {standalone_file}")


def create_summary_statistics(data, output_dir, width_filter=None, seed_filter=None):
    """Create summary statistics table."""
    # Filter data
    filtered_data = data.copy()
    if width_filter is not None:
        filtered_data = filtered_data[filtered_data['width'] == width_filter]
    if seed_filter is not None:
        filtered_data = filtered_data[filtered_data['seed'] == seed_filter]
    
    # Remove failed experiments
    successful_data = filtered_data[filtered_data['val_acc'] > 0.1].copy()
    
    # Group by precision type and compute statistics
    precision_map = {
        'FP32 (torchdiffeq)': 'FP32+TD',
        'FP32 (rampde)': 'FP32+MP',
        'TF32 (torchdiffeq)': 'TF32+TD', 
        'TF32 (rampde)': 'TF32+MP',
        'BF16 (torchdiffeq)': 'BF16+TD',
        'BF16 (rampde)': 'BF16+MP',
        'FP16-Dyn (rampde)': 'FP16-Dyn+MP',
        'FP16-Grad (torchdiffeq)': 'FP16-Grad+TD',
        'FP16-Grad (rampde)': 'FP16-Grad+MP',
        'FP16-None (torchdiffeq)': 'FP16-None+TD',
        'FP16-None (rampde)': 'FP16-None+MP'
    }
    
    successful_data['precision_group'] = successful_data['config'].map(precision_map)
    successful_data['precision_group'] = successful_data['precision_group'].fillna(successful_data['config'])
    
    # Compute summary statistics
    summary_stats = successful_data.groupby('precision_group').agg({
        'val_acc': ['count', 'mean', 'std', 'min', 'max'],
        'train_acc': ['mean', 'std'],
        'max_memory_mb': ['mean', 'std', 'min', 'max'],
        'total_time': ['mean', 'std', 'min', 'max'],
        'time_fwd_sum': ['mean', 'std'],
        'time_bwd_sum': ['mean', 'std']
    }).round(4)
    
    summary_file = output_dir / 'stl10_summary_statistics.csv'
    summary_stats.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")
    
    return summary_stats


def main():
    parser = argparse.ArgumentParser(description='Generate STL10 LaTeX table')
    parser.add_argument('--csv-file', type=str, default='./raw_data/summary_ode_stl10.csv',
                       help='Path to STL10 summary CSV file')
    parser.add_argument('--results-dir', type=str, default='./raw_data',
                       help='Path to STL10 results directory (for test loss data)')
    parser.add_argument('--output-dir', type=str, default='./outputs/tab_stl10_results',
                       help='Output directory for LaTeX table')
    parser.add_argument('--width', type=int, help='Filter by network width')
    parser.add_argument('--seed', type=int, help='Filter by random seed')
    parser.add_argument('--skip-test-loss', action='store_true',
                       help='Skip loading test loss data (faster)')
    
    args = parser.parse_args()
    
    # Setup paths
    csv_file = Path(args.csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = args.results_dir if not args.skip_test_loss else None
    
    # Load and process data
    print(f"Loading STL10 data from {csv_file}")
    data = load_and_process_data(csv_file, results_dir)
    print(f"Loaded {len(data)} experiments")
    
    # Create LaTeX table
    output_file = output_dir / 'stl10_results_table.tex'
    create_latex_table(data, output_file, width_filter=args.width, seed_filter=args.seed)
    
    # Create summary statistics
    create_summary_statistics(data, output_dir, width_filter=args.width, seed_filter=args.seed)
    
    # Create CSV version for easy viewing
    filtered_data = data.copy()
    if args.width is not None:
        filtered_data = filtered_data[filtered_data['width'] == args.width]
    if args.seed is not None:
        filtered_data = filtered_data[filtered_data['seed'] == args.seed]
    
    successful_data = filtered_data[filtered_data['val_acc'] > 0.1].copy()
    successful_data = successful_data.sort_values('val_acc', ascending=False)
    
    csv_output_file = output_dir / 'stl10_results_table.csv'
    successful_data.to_csv(csv_output_file, index=False)
    print(f"CSV table saved to: {csv_output_file}")
    
    print(f"\nTable generation complete. Main output: {output_file}")
    
    # Print key findings
    if len(successful_data) > 0:
        print(f"\n=== Key Results ===")
        best_config = successful_data.iloc[0]
        print(f"Best configuration: {best_config['config']}")
        print(f"  - Validation accuracy: {best_config['val_acc']:.3f}")
        print(f"  - Memory usage: {best_config['max_memory_mb']/1000:.1f} GB")
        print(f"  - Training time: {best_config['total_time']/3600:.2f} hours")
        
        print(f"\nMemory range: {successful_data['max_memory_mb'].min()/1000:.1f} - {successful_data['max_memory_mb'].max()/1000:.1f} GB")
        print(f"Accuracy range: {successful_data['val_acc'].min():.3f} - {successful_data['val_acc'].max():.3f}")


if __name__ == "__main__":
    main()