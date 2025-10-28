#!/usr/bin/env python3
"""
Aggregate STL10 experiment results into a summary CSV.

This script processes all STL10 experiment directories and extracts key metrics
from the final epoch to create a summary table.
"""

import pandas as pd
from pathlib import Path
import argparse
import json


def parse_experiment_name(dirname):
    """Parse experiment directory name to extract parameters."""
    # Format: stl10_{precision}_{scaler}_{solver}_rk4_...
    # Example: stl10_bfloat16_rampde_rk4_stable_stable_lr_0.05_nepochs_5_batch_size_16_width_128_seed26_20250930_122411

    parts = dirname.split('_')

    if len(parts) < 3 or not dirname.startswith('stl10_'):
        return None

    dataset = parts[0]  # stl10
    precision = parts[1]  # float16, bfloat16, tfloat32

    # Float16 has scaler in position 2, solver in position 3
    # Others have solver in position 2
    if precision == "float16":
        if len(parts) < 4:
            return None
        scaler = parts[2]  # grad, dynamic, or none
        solver = parts[3]  # torchdiffeq or rampde
    else:
        if len(parts) < 3:
            return None
        solver = parts[2]  # torchdiffeq or rampde
        scaler = "none"

    # Extract seed and width from directory name
    seed = None
    width = None
    for i, part in enumerate(parts):
        if part.startswith('seed'):
            try:
                seed = int(part[4:])  # Remove 'seed' prefix
            except ValueError:
                pass
        if part == 'width' and i + 1 < len(parts):
            try:
                width = int(parts[i + 1])
            except ValueError:
                pass

    return {
        'directory': dirname,
        'data_name': dataset,
        'precision_str': precision,
        'odeint_type': solver,
        'scaler_name': scaler,
        'seed': seed,
        'width': width
    }


def extract_metrics_from_csv(csv_path):
    """Extract final metrics from experiment CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None

        # Get the last row (final epoch)
        final = df.iloc[-1]

        metrics = {
            'val_acc': final['val_acc'] if 'val_acc' in final else None,
            'train_acc': final['train_acc'] if 'train_acc' in final else None,
            'val_loss': final['val_loss'] if 'val_loss' in final else None,
            'train_loss': final['train_loss'] if 'train_loss' in final else None,
            'time_fwd': final['time_fwd'] if 'time_fwd' in final else None,
            'time_bwd': final['time_bwd'] if 'time_bwd' in final else None,
            'time_fwd_sum': final['time_fwd_sum'] if 'time_fwd_sum' in final else None,
            'time_bwd_sum': final['time_bwd_sum'] if 'time_bwd_sum' in final else None,
            'max_memory_mb': final['max_memory_mb'] if 'max_memory_mb' in final else None,
            'epoch': int(final['epoch']) if 'epoch' in final else None,
            'iter': int(final['iter']) if 'iter' in final else None
        }

        # Try to read additional metadata from args.csv if available
        args_csv = csv_path.parent / 'args.csv'
        if args_csv.exists():
            args_df = pd.read_csv(args_csv)
            if len(args_df) > 0:
                args_row = args_df.iloc[0]
                for col in ['method', 'gpu', 'timestamp', 'job_id', 'tol', 'nepochs',
                           'lr', 'momentum', 'batch_size', 'test_batch_size', 'weight_decay',
                           'precision', 'odeint', 'unstable', 'no_grad_scaler', 'no_dynamic_scaler',
                           'results_dir', 'debug', 'test_freq', 'stable']:
                    if col in args_row:
                        metrics[col] = args_row[col]

        return metrics
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def aggregate_stl10_results(raw_data_dir, seed_filter=None, width_filter=None):
    """Aggregate all STL10 experiment results into a summary DataFrame.

    Args:
        raw_data_dir: Directory containing experiment subdirectories
        seed_filter: Optional seed number to filter results
        width_filter: Optional width to filter results
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

        if not dirname.startswith('stl10_'):
            continue

        # Parse experiment parameters
        params = parse_experiment_name(dirname)
        if params is None:
            print(f"Skipping {dirname} - could not parse")
            continue

        # Filter by seed if specified
        if seed_filter is not None and params['seed'] != seed_filter:
            continue

        # Filter by width if specified
        if width_filter is not None and params['width'] != width_filter:
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

        scaler_str = f" - {params['scaler_name']}" if params['scaler_name'] != 'none' else ""
        print(f"Processed: {params['data_name']} - {params['precision_str']} - {params['odeint_type']}{scaler_str} (seed={params['seed']}, width={params['width']})")

    if len(results) == 0:
        print("No results found!")
        return None

    df = pd.DataFrame(results)

    # Sort by precision, solver, scaler
    precision_order = ['float32', 'tfloat32', 'bfloat16', 'float16']
    scaler_order = ['none', 'grad', 'dynamic']

    df['precision_order'] = df['precision_str'].map({p: i for i, p in enumerate(precision_order)})
    df['scaler_order'] = df['scaler_name'].map({s: i for i, s in enumerate(scaler_order)})

    df = df.sort_values(['precision_order', 'odeint_type', 'scaler_order', 'seed', 'width'])
    df = df.drop(['precision_order', 'scaler_order'], axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(description='Aggregate STL10 experiment results')
    parser.add_argument('--raw-data-dir', type=str, default='raw_data/ode_stl10',
                       help='Directory containing raw experiment data')
    parser.add_argument('--output', type=str, default='raw_data/ode_stl10/summary_ode_stl10.csv',
                       help='Output CSV file path')
    parser.add_argument('--seed', type=int, default=None,
                       help='Filter results by random seed')
    parser.add_argument('--width', type=int, default=None,
                       help='Filter results by network width')
    args = parser.parse_args()

    print("Aggregating STL10 experiment results...")
    print(f"Raw data directory: {args.raw_data_dir}")
    if args.seed is not None:
        print(f"Filtering by seed: {args.seed}")
    if args.width is not None:
        print(f"Filtering by width: {args.width}")

    df = aggregate_stl10_results(args.raw_data_dir, seed_filter=args.seed, width_filter=args.width)

    if df is not None:
        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"\nSaved summary to: {args.output}")
        print(f"Total experiments: {len(df)}")

        # Print summary
        print("\nSummary by configuration:")
        summary = df.groupby(['data_name', 'precision_str', 'odeint_type', 'scaler_name']).size()
        print(summary)
    else:
        print("Failed to aggregate results")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
