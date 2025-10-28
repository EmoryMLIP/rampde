#!/usr/bin/env python3
"""
Aggregate CNF experiment results into a summary CSV.

This script processes all CNF experiment directories and extracts key metrics
from the final iteration to create a summary table.
"""

import pandas as pd
import re
from pathlib import Path
import argparse


def parse_experiment_name(dirname):
    """Parse experiment directory name to extract parameters."""
    # Format:
    # - float32/bfloat16/tfloat32: {dataset}_{precision}_{solver}_rk4_...
    # - float16: {dataset}_{precision}_{scaler}_{solver}_rk4_...
    parts = dirname.split('_')

    if len(parts) < 3:
        return None

    dataset = parts[0]
    precision = parts[1]

    # Float16 has scaler in position 2, solver in position 3
    if precision == "float16":
        if len(parts) < 4:
            return None
        scaler = parts[2]  # grad, dynamic, or none
        solver = parts[3]  # torchdiffeq or rampde
    else:
        # Float32, bfloat16, tfloat32: solver is in position 2
        solver = parts[2]
        scaler = "none"

    return {
        'dataset': dataset,
        'precision_str': precision,
        'odeint_type': solver,
        'scaler_name': scaler
    }


def extract_metrics_from_csv(csv_path):
    """Extract final metrics from experiment CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None

        # Get the last row (final iteration)
        final = df.iloc[-1]

        return {
            'val_loss': final['val_loss'],
            'time_fwd': final['time_fwd'],
            'time_bwd': final['time_bwd'],
            'time_fwd_sum': final['time_fwd_sum'],
            'time_bwd_sum': final['time_bwd_sum'],
            'max_memory_mb': final['max_memory_mb'],
            'final_iter': int(final['iter'])
        }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def aggregate_cnf_results(raw_data_dir, seed_filter=None):
    """Aggregate all CNF experiment results into a summary DataFrame.

    Args:
        raw_data_dir: Directory containing experiment subdirectories
        seed_filter: Optional seed number to filter results (e.g., 24 or 42)
    """
    raw_data_path = Path(raw_data_dir)

    if not raw_data_path.exists():
        print(f"Error: Directory {raw_data_dir} does not exist")
        return None

    results = []

    for exp_dir in sorted(raw_data_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        dirname = exp_dir.name

        # Filter by seed if specified
        if seed_filter is not None:
            if f"_seed{seed_filter}_" not in dirname:
                continue

        # Parse experiment parameters
        params = parse_experiment_name(dirname)
        if params is None:
            print(f"Skipping {dirname} - could not parse")
            continue

        # Find the CSV file (exclude args.csv)
        csv_files = [f for f in exp_dir.glob('*.csv') if f.name != 'args.csv']
        if len(csv_files) == 0:
            print(f"Warning: No data CSV file found in {dirname}")
            continue

        csv_path = csv_files[0]

        # Extract metrics
        metrics = extract_metrics_from_csv(csv_path)
        if metrics is None:
            continue

        # Combine parameters and metrics
        result = {**params, **metrics}
        results.append(result)

        print(f"Processed: {params['dataset']} - {params['precision_str']} - {params['odeint_type']} - {params['scaler_name']}")

    if len(results) == 0:
        print("No results found!")
        return None

    df = pd.DataFrame(results)

    # Rename dataset to data_name for compatibility with table generation script
    df['data_name'] = df['dataset']

    # Sort by dataset, then precision, then solver, then scaler
    precision_order = ['float32', 'tfloat32', 'bfloat16', 'float16']
    scaler_order = ['none', 'grad', 'dynamic']

    df['precision_order'] = df['precision_str'].map({p: i for i, p in enumerate(precision_order)})
    df['scaler_order'] = df['scaler_name'].map({s: i for i, s in enumerate(scaler_order)})

    df = df.sort_values(['dataset', 'precision_order', 'odeint_type', 'scaler_order'])
    df = df.drop(['precision_order', 'scaler_order'], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(description='Aggregate CNF experiment results')
    parser.add_argument('--raw-data-dir', type=str, default='raw_data/cnf',
                       help='Directory containing raw experiment data')
    parser.add_argument('--output', type=str, default='raw_data/cnf/summary_cnf.csv',
                       help='Output CSV file path')
    parser.add_argument('--seed', type=int, default=None,
                       help='Filter results by random seed (e.g., 24 or 42)')
    args = parser.parse_args()

    print("Aggregating CNF experiment results...")
    print(f"Raw data directory: {args.raw_data_dir}")
    if args.seed is not None:
        print(f"Filtering by seed: {args.seed}")

    df = aggregate_cnf_results(args.raw_data_dir, seed_filter=args.seed)

    if df is not None:
        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"\nSaved summary to: {args.output}")
        print(f"Total experiments: {len(df)}")

        # Print summary
        print("\nSummary by configuration:")
        summary = df.groupby(['dataset', 'precision_str', 'scaler_name']).size()
        print(summary)
    else:
        print("Failed to aggregate results")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
