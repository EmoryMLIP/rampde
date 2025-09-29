#!/usr/bin/env python3
"""
Universal Result Collection Script for rampde experiments.

This script scans experiment directories and collects results into a unified DataFrame.
It supports filtering by experiment type, date, precision, method, and other parameters.
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

# Standardized column mappings for backward compatibility
COLUMN_MAPPINGS = {
    # Iteration/epoch columns
    'step': 'iter',
    'iteration': 'iter',
    
    # Learning rate
    'learning rate': 'lr',
    
    # Loss columns
    'running_NLL': 'running_loss',
    'val_NLL': 'val_loss',
    'test_loss': 'val_loss',
    
    # Accuracy columns  
    'test_acc': 'val_acc',
    
    # Time columns
    'time': 'time_fwd',  # For experiments with only total time
    'time_avg': 'time_fwd',  # For experiments with only average time
    'batch_time': 'time_fwd',
    'batch_time_avg': 'time_fwd',
    'step_time': 'time_fwd',
    'time fwd': 'time_fwd',
    'time bwd': 'time_bwd',
    
    # Memory columns
    'max_memory': 'max_memory_mb',
    'max memory (MB)': 'max_memory_mb',
    'mem_peak_MB': 'max_memory_mb',
    'peak_memory_mb': 'max_memory_mb',
    
    # NFE columns (to be removed)
    'f_nfe': None,
    'b_nfe': None,
    'nfe_fwd': None,
    'nfe_bwd': None,
    'nfe fwd': None,
    'nfe bwd': None,
}

def parse_folder_name(folder_name):
    """Parse experiment folder name to extract metadata."""
    parts = folder_name.split('_')
    metadata = {}
    
    # Common patterns across experiments
    if len(parts) >= 4:
        metadata['dataset'] = parts[0]
        metadata['precision'] = parts[1]
        
        # Handle precision+scaler formats like "float16_grad"
        if len(parts) > 2 and parts[2] in ['grad', 'dynamic', 'none']:
            metadata['precision'] = f"{parts[1]}_{parts[2]}"
            metadata['odeint'] = parts[3] if len(parts) > 3 else 'unknown'
            metadata['method'] = parts[4] if len(parts) > 4 else 'unknown'
        else:
            metadata['odeint'] = parts[2] if len(parts) > 2 else 'unknown'
            metadata['method'] = parts[3] if len(parts) > 3 else 'unknown'
    
    # Extract seed if present
    for part in parts:
        if part.startswith('seed'):
            metadata['seed'] = part.replace('seed', '')
            break
    
    # Extract timestamp (last part that looks like a timestamp)
    for part in reversed(parts):
        if len(part) == 15 and part.isdigit():  # YYYYMMDD_HHMMSS format
            try:
                metadata['timestamp'] = datetime.strptime(part, '%Y%m%d_%H%M%S')
            except:
                pass
            break
    
    return metadata

def read_experiment_data(result_dir):
    """Read experiment data from a single result directory."""
    data = {}
    folder_name = os.path.basename(result_dir)
    
    # Parse folder name for metadata
    metadata = parse_folder_name(folder_name)
    data['folder_name'] = folder_name
    data['result_dir'] = result_dir
    data.update(metadata)
    
    # Read args.csv if it exists
    args_path = os.path.join(result_dir, 'args.csv')
    if os.path.exists(args_path):
        try:
            args_df = pd.read_csv(args_path)
            if len(args_df) > 0:
                for col in args_df.columns:
                    data[f'arg_{col}'] = args_df[col].iloc[0]
        except Exception as e:
            print(f"Warning: Could not read args.csv in {result_dir}: {e}")
    
    # Read metrics CSV (folder_name.csv)
    csv_path = os.path.join(result_dir, f"{folder_name}.csv")
    if os.path.exists(csv_path):
        try:
            metrics_df = pd.read_csv(csv_path)
            
            # Rename columns according to mapping
            rename_dict = {}
            for old_col in metrics_df.columns:
                if old_col in COLUMN_MAPPINGS:
                    new_col = COLUMN_MAPPINGS[old_col]
                    if new_col is not None:  # Skip columns mapped to None
                        rename_dict[old_col] = new_col
            
            metrics_df = metrics_df.rename(columns=rename_dict)
            
            # Drop NFE columns and any columns mapped to None
            drop_cols = [col for col, new_col in COLUMN_MAPPINGS.items() if new_col is None]
            metrics_df = metrics_df.drop(columns=[col for col in drop_cols if col in metrics_df.columns])
            
            # Extract final metrics (last row)
            if len(metrics_df) > 0:
                final_row = metrics_df.iloc[-1]
                for col in final_row.index:
                    data[f'final_{col}'] = final_row[col]
                
                # Also store best validation accuracy if available
                if 'val_acc' in metrics_df.columns:
                    data['best_val_acc'] = metrics_df['val_acc'].max()
                
                # Store training duration info
                data['num_iterations'] = len(metrics_df)
                data['final_iter'] = metrics_df['iter'].iloc[-1] if 'iter' in metrics_df.columns else len(metrics_df)
                
                # Calculate average times if available
                if 'time_fwd' in metrics_df.columns:
                    data['avg_time_fwd'] = metrics_df['time_fwd'].mean()
                if 'time_bwd' in metrics_df.columns:
                    data['avg_time_bwd'] = metrics_df['time_bwd'].mean()
                if 'max_memory_mb' in metrics_df.columns:
                    data['peak_memory_mb'] = metrics_df['max_memory_mb'].max()
            
            data['has_metrics'] = True
            data['metrics_rows'] = len(metrics_df)
        except Exception as e:
            print(f"Warning: Could not read metrics CSV in {result_dir}: {e}")
            data['has_metrics'] = False
    else:
        data['has_metrics'] = False
    
    # Check for completion status
    data['has_final_model'] = os.path.exists(os.path.join(result_dir, 'ckpt.pth')) or \
                              os.path.exists(os.path.join(result_dir, 'ckpt_final.pth'))
    data['has_emergency_stop'] = os.path.exists(os.path.join(result_dir, 'ckpt_emergency_stop.pth'))
    
    return data

def collect_all_results(base_dir, experiment_filter=None, date_filter=None, 
                       precision_filter=None, method_filter=None):
    """Collect results from all experiment directories."""
    results = []
    
    # Define experiment directories to scan
    experiment_dirs = ['ode_mnist', 'ode_stl10', 'cnf', 'otflow', 'otflowlarge', 'ode']
    
    if experiment_filter:
        experiment_dirs = [d for d in experiment_dirs if d in experiment_filter]
    
    for exp_type in experiment_dirs:
        exp_dir = os.path.join(base_dir, exp_type)
        if not os.path.exists(exp_dir):
            continue
            
        # Find all result directories
        result_dirs = [d for d in glob.glob(os.path.join(exp_dir, '*')) 
                      if os.path.isdir(d) and not os.path.basename(d).startswith('.')]
        
        for result_dir in result_dirs:
            # Apply filters
            folder_name = os.path.basename(result_dir)
            
            # Date filter
            if date_filter:
                folder_metadata = parse_folder_name(folder_name)
                if 'timestamp' not in folder_metadata:
                    continue
                if folder_metadata['timestamp'] < date_filter:
                    continue
            
            # Precision filter
            if precision_filter:
                if not any(p in folder_name for p in precision_filter):
                    continue
            
            # Method filter  
            if method_filter:
                if not any(m in folder_name for m in method_filter):
                    continue
            
            # Read experiment data
            exp_data = read_experiment_data(result_dir)
            exp_data['experiment_type'] = exp_type
            results.append(exp_data)
    
    return pd.DataFrame(results)

def print_summary_stats(df):
    """Print summary statistics for the collected results."""
    print("\n=== Summary Statistics ===")
    print(f"Total experiments found: {len(df)}")
    print(f"Experiments with metrics: {df['has_metrics'].sum()}")
    print(f"Completed experiments: {df['has_final_model'].sum()}")
    print(f"Emergency stops: {df['has_emergency_stop'].sum()}")
    
    print("\n=== Experiments by Type ===")
    print(df['experiment_type'].value_counts())
    
    print("\n=== Experiments by Precision ===")
    if 'precision' in df.columns:
        print(df['precision'].value_counts())
    
    print("\n=== Experiments by ODE Integrator ===")
    if 'odeint' in df.columns:
        print(df['odeint'].value_counts())
    
    print("\n=== Experiments by Method ===")
    if 'method' in df.columns:
        print(df['method'].value_counts())

def create_comparison_table(df, group_by=['experiment_type', 'dataset', 'precision', 'odeint', 'method'],
                          metrics=['final_val_acc', 'final_val_loss', 'avg_time_fwd', 'avg_time_bwd', 'peak_memory_mb']):
    """Create a comparison table of experiments."""
    # Filter to only experiments with metrics
    df_metrics = df[df['has_metrics'] == True].copy()
    
    if len(df_metrics) == 0:
        print("No experiments with metrics found.")
        return None
    
    # Select available group_by columns
    available_group_by = [col for col in group_by if col in df_metrics.columns]
    if not available_group_by:
        print("No grouping columns available.")
        return None
    
    # Select available metrics
    available_metrics = [col for col in metrics if col in df_metrics.columns]
    if not available_metrics:
        print("No metrics columns available.")
        return None
    
    # Group and aggregate
    agg_dict = {metric: ['mean', 'std', 'count'] for metric in available_metrics}
    comparison = df_metrics.groupby(available_group_by).agg(agg_dict)
    
    # Flatten column names
    comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description='Collect and analyze rampde experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Base results directory')
    parser.add_argument('--output', type=str, default='experiment_summary.csv',
                       help='Output CSV file for collected results')
    parser.add_argument('--experiment', type=str, nargs='+',
                       help='Filter by experiment type (ode_mnist, ode_stl10, cnf, otflow, otflowlarge, ode, etc.)')
    parser.add_argument('--days', type=int,
                       help='Only include experiments from the last N days')
    parser.add_argument('--precision', type=str, nargs='+',
                       help='Filter by precision (float32, float16, etc.)')
    parser.add_argument('--method', type=str, nargs='+',
                       help='Filter by method (rk4, euler, etc.)')
    parser.add_argument('--comparison', type=str,
                       help='Output file for comparison table')
    parser.add_argument('--json', type=str,
                       help='Output file for JSON export')
    
    args = parser.parse_args()
    
    # Set up date filter
    date_filter = None
    if args.days:
        date_filter = datetime.now() - timedelta(days=args.days)
    
    # Collect results
    print(f"Scanning results directory: {args.results_dir}")
    df = collect_all_results(
        args.results_dir,
        experiment_filter=args.experiment,
        date_filter=date_filter,
        precision_filter=args.precision,
        method_filter=args.method
    )
    
    if len(df) == 0:
        print("No experiments found matching the filters.")
        return
    
    # Print summary statistics
    print_summary_stats(df)
    
    # Save full results
    df.to_csv(args.output, index=False)
    print(f"\nFull results saved to: {args.output}")
    
    # Create comparison table if requested
    if args.comparison:
        comparison = create_comparison_table(df)
        if comparison is not None:
            comparison.to_csv(args.comparison)
            print(f"Comparison table saved to: {args.comparison}")
            print("\n=== Top Performers by Validation Accuracy ===")
            if 'final_val_acc_mean' in comparison.columns:
                print(comparison.sort_values('final_val_acc_mean', ascending=False).head(10))
    
    # Export to JSON if requested
    if args.json:
        # Convert datetime objects to strings for JSON serialization
        df_json = df.copy()
        for col in df_json.columns:
            if df_json[col].dtype == 'datetime64[ns]':
                df_json[col] = df_json[col].astype(str)
        
        df_json.to_json(args.json, orient='records', indent=2)
        print(f"JSON export saved to: {args.json}")
    
    # Print experiments with best validation accuracy by type
    if 'final_val_acc' in df.columns and 'experiment_type' in df.columns:
        print("\n=== Best Validation Accuracy by Experiment Type ===")
        for exp_type in df['experiment_type'].unique():
            exp_df = df[df['experiment_type'] == exp_type]
            if 'final_val_acc' in exp_df.columns and exp_df['final_val_acc'].notna().any():
                best_idx = exp_df['final_val_acc'].idxmax()
                best_exp = exp_df.loc[best_idx]
                print(f"\n{exp_type}: {best_exp['final_val_acc']:.4f}")
                print(f"  Folder: {best_exp['folder_name']}")
                if 'avg_time_fwd' in best_exp:
                    print(f"  Avg forward time: {best_exp['avg_time_fwd']:.4f}s")
                if 'peak_memory_mb' in best_exp:
                    print(f"  Peak memory: {best_exp['peak_memory_mb']:.1f}MB")

if __name__ == '__main__':
    main()