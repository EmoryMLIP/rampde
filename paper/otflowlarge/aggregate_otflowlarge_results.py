#!/usr/bin/env python3
"""
Aggregate OTFlowLarge experiment results into a summary CSV.

This script processes all OTFlowLarge experiment directories and extracts key metrics
from the final iteration to create a summary table.
"""

import pandas as pd
from pathlib import Path
import argparse


def parse_experiment_name(dirname):
    """Parse experiment directory name to extract parameters."""
    # Format: bsds300_{precision}_{scaler}_{solver}_rk4_alpha_...
    # Example: bsds300_float16_grad_rampde_rk4_alpha_1.0,2000.0,800.0_lr_0.001_niters_10000_batch_size_512_hidden_dim_1024_nt_16_seed42_20251006_090038

    parts = dirname.split('_')

    if len(parts) < 3:
        return None

    dataset = parts[0]  # bsds300, power, gas, etc.
    precision = parts[1]  # float16, bfloat16, tfloat32, float32

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
        scaler = ""  # empty for non-float16

    # Extract seed from directory name
    seed = None
    for i, part in enumerate(parts):
        if part.startswith('seed'):
            try:
                seed = int(part[4:])  # Remove 'seed' prefix
            except ValueError:
                pass

    return {
        'directory': dirname,
        'data_name': dataset,
        'precision_str': precision,
        'odeint_type': solver,
        'scaler_name': scaler,
        'seed': seed,
    }


def extract_metrics_from_csv(csv_path):
    """Extract final metrics from experiment CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None

        # Get the last row (final iteration)
        final = df.iloc[-1]

        metrics = {
            'val_loss': final['val_loss'] if 'val_loss' in final else None,
            'val_L': final['val_L'] if 'val_L' in final else None,
            'val_NLL': final['val_NLL'] if 'val_NLL' in final else None,
            'val_HJB': final['val_HJB'] if 'val_HJB' in final else None,
            'val_mmd': final['val_mmd'] if 'val_mmd' in final else None,
            'time_fwd': final['time_fwd'] if 'time_fwd' in final else None,
            'time_bwd': final['time_bwd'] if 'time_bwd' in final else None,
            'time_fwd_sum': final['time_fwd_sum'] if 'time_fwd_sum' in final else None,
            'time_bwd_sum': final['time_bwd_sum'] if 'time_bwd_sum' in final else None,
            'max_memory_mb': final['max_memory_mb'] if 'max_memory_mb' in final else None,
            'iter': int(final['iter']) if 'iter' in final else None
        }

        # Try to read additional metadata from args.csv if available
        args_csv = csv_path.parent / 'args.csv'
        if args_csv.exists():
            args_df = pd.read_csv(args_csv)
            if len(args_df) > 0:
                args_row = args_df.iloc[0]
                for col in ['method', 'gpu', 'timestamp', 'job_id',
                           'niters', 'lr', 'batch_size', 'test_batch_size',
                           'precision', 'odeint', 'no_grad_scaler', 'no_dynamic_scaler',
                           'results_dir', 'debug', 'val_freq', 'm', 'nt', 'nt_val',
                           'alph', 'weight_decay', 'early_stopping']:
                    if col in args_row:
                        metrics[col] = args_row[col]

        return metrics

    except Exception as e:
        print(f"Warning: Error extracting metrics from {csv_path}: {e}")
        return None


def aggregate_results(raw_data_dir, output_csv, seed_filter=None):
    """Aggregate all experiment results into a summary CSV."""

    raw_data_path = Path(raw_data_dir)

    if not raw_data_path.exists():
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        return False

    # Find all experiment directories
    experiment_dirs = [d for d in raw_data_path.iterdir() if d.is_dir()]

    if not experiment_dirs:
        print(f"Error: No experiment directories found in {raw_data_dir}")
        return False

    print(f"Found {len(experiment_dirs)} experiment directories")

    all_results = []

    for exp_dir in sorted(experiment_dirs):
        # Parse experiment name
        exp_info = parse_experiment_name(exp_dir.name)

        if exp_info is None:
            print(f"Warning: Could not parse experiment name: {exp_dir.name}")
            continue

        # Apply seed filter if specified
        if seed_filter is not None and exp_info['seed'] != seed_filter:
            continue

        # Find the results CSV file (should match directory name)
        csv_file = exp_dir / f"{exp_dir.name}.csv"

        if not csv_file.exists():
            print(f"Warning: Results CSV not found for {exp_dir.name}")
            continue

        # Extract metrics
        metrics = extract_metrics_from_csv(csv_file)

        if metrics is None:
            print(f"Warning: Could not extract metrics from {exp_dir.name}")
            continue

        # Combine experiment info and metrics
        result = {**exp_info, **metrics}
        all_results.append(result)

        print(f"  ✓ Processed {exp_dir.name}")

    if not all_results:
        print("Error: No valid experiment results found")
        return False

    # Create DataFrame and save
    df = pd.DataFrame(all_results)

    # Sort by key columns for consistent ordering
    sort_cols = [col for col in ['data_name', 'precision_str', 'scaler_name', 'odeint_type', 'seed']
                 if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n✓ Summary CSV created: {output_csv}")
    print(f"  Total experiments: {len(df)}")
    if 'seed' in df.columns:
        print(f"  Seeds: {sorted(df['seed'].unique())}")
    if 'precision_str' in df.columns:
        print(f"  Precisions: {sorted(df['precision_str'].unique())}")
    if 'odeint_type' in df.columns:
        print(f"  Solvers: {sorted(df['odeint_type'].unique())}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Aggregate OTFlowLarge experiment results')
    parser.add_argument('--raw-data-dir', default='./raw_data/otflowlarge',
                       help='Directory containing experiment subdirectories')
    parser.add_argument('--output', default='./raw_data/otflowlarge/summary_otflowlarge.csv',
                       help='Output CSV file path')
    parser.add_argument('--seed', type=int, default=None,
                       help='Filter results by seed (default: include all seeds)')

    args = parser.parse_args()

    print("=" * 60)
    print("OTFlowLarge Results Aggregation")
    print("=" * 60)
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Output file: {args.output}")
    if args.seed is not None:
        print(f"Seed filter: {args.seed}")
    print()

    success = aggregate_results(args.raw_data_dir, args.output, args.seed)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
