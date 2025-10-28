#!/usr/bin/env python3
"""
Plot roundoff experiment results comparing rampde vs torchdiffeq
for RK4 and Euler methods with configurable precision and scaler.
Shows all error metrics with distinct line styles.
Supports CNF, STL10, and OTFLOW experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def get_error_columns(experiment):
    """Get appropriate error columns for each experiment type."""
    if experiment == 'cnf':
        return {
            'columns': [
                'sol_error_mean',
                'grad_y0_error_mean', 
                'grad_hyper_net.fc1.weight_error_mean',
                'grad_hyper_net.fc1.bias_error_mean',
                'grad_hyper_net.fc2.weight_error_mean',
                'grad_hyper_net.fc2.bias_error_mean',
                'grad_hyper_net.fc3.weight_error_mean',
                'grad_hyper_net.fc3.bias_error_mean'
            ],
            'labels': [
                'Solution Error',
                'Grad y0 Error',
                'Grad FC1 Weight Error',
                'Grad FC1 Bias Error', 
                'Grad FC2 Weight Error',
                'Grad FC2 Bias Error',
                'Grad FC3 Weight Error',
                'Grad FC3 Bias Error'
            ]
        }
    elif experiment == 'otflow':
        return {
            'columns': [
                'sol_error_mean',
                'grad_y0_error_mean',
                'grad_Phi.A_error_mean',
                'grad_Phi.c.weight_error_mean',
                'grad_Phi.c.bias_error_mean',
                'grad_Phi.w.weight_error_mean',
                'grad_Phi.N.layers.0.weight_error_mean',
                'grad_Phi.N.layers.0.bias_error_mean'
            ],
            'labels': [
                'Solution Error',
                'Grad y0 Error',
                'Grad Phi A Error',
                'Grad Phi c Weight Error',
                'Grad Phi c Bias Error',
                'Grad Phi w Weight Error',
                'Grad Phi N Layer0 Weight Error',
                'Grad Phi N Layer0 Bias Error'
            ]
        }
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")

def main():
    parser = argparse.ArgumentParser(description='Plot roundoff experiment results')
    parser.add_argument('--experiment', type=str, default='cnf', 
                       choices=['cnf', 'stl10', 'otflow'],
                       help='Experiment type (default: cnf)')
    parser.add_argument('--precision', type=str, default='bfloat16',
                       help='Precision to filter for (default: bfloat16)')
    parser.add_argument('--scaler', type=str, default=None,
                       help='Scaler type to filter for (default: None, shows all)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result CSV files (default: results)')
    args = parser.parse_args()

    # Read the CSV file - try complete version first
    csv_file = os.path.join(args.results_dir, f'{args.experiment}_roundoff_results_complete.csv')
    if not os.path.exists(csv_file):
        csv_file = os.path.join(args.results_dir, f'{args.experiment}_roundoff_results.csv')
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found: {csv_file}")
            return
    
    df = pd.read_csv(csv_file)

    # Filter for specified precision
    df_filtered = df[df['precision'] == args.precision].copy()
    
    # Filter for scaler if specified
    if args.scaler is not None:
        df_filtered = df_filtered[df_filtered['scaler_type'] == args.scaler].copy()
    
    if df_filtered.empty:
        print(f"No data found for precision={args.precision}, scaler={args.scaler}")
        return

    print(f"Plotting data for {args.experiment} experiment:")
    print(f"  precision={args.precision}, scaler={args.scaler}")
    print(f"  Found {len(df_filtered)} rows of data")

    # Get error columns and labels for this experiment type
    error_config = get_error_columns(args.experiment)
    error_columns = error_config['columns']
    error_labels = error_config['labels']

    # Define line styles for distinction
    line_styles = [
        ('o-', 'blue'),      # Solution
        ('s-', 'red'),       # Grad y0
        ('^-', 'green'),     # FC1 weight
        ('v-', 'orange'),    # FC1 bias
        ('d-', 'purple'),    # FC2 weight
        ('p-', 'brown'),     # FC2 bias
        ('*-', 'pink'),      # FC3 weight
        ('h-', 'gray')       # FC3 bias
    ]

    # Create figure with 2x2 subplots + legend subplot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.3])

    axes = [
        fig.add_subplot(gs[0, 0]),  # rampde RK4
        fig.add_subplot(gs[0, 1]),  # torchdiffeq RK4  
        fig.add_subplot(gs[1, 0]),  # rampde Euler
        fig.add_subplot(gs[1, 1])   # torchdiffeq Euler
    ]

    legend_ax = fig.add_subplot(gs[:, 2])

    # Create title based on configuration
    title_parts = [f'{args.experiment.upper()} Roundoff Error Analysis: {args.precision} Precision']
    if args.scaler is not None:
        title_parts.append(f'Scaler: {args.scaler}')
    fig.suptitle(' - '.join(title_parts), fontsize=16)

    # Extract unique timestep values for x-axis
    timesteps = sorted(df_filtered['n_timesteps'].unique())

    # Plot configurations
    plot_configs = [
        ('rampde', 'rk4', 'rampde - RK4'),
        ('torchdiffeq', 'rk4', 'torchdiffeq - RK4'),
        ('rampde', 'euler', 'rampde - Euler'),
        ('torchdiffeq', 'euler', 'torchdiffeq - Euler')
    ]

    # Collect all error values to determine global y-axis limits
    all_error_values = []
    plot_data = []

    for i, (odeint_type, method, title) in enumerate(plot_configs):
        # Filter data for this configuration
        data_subset = df_filtered[(df_filtered['odeint_type'] == odeint_type) & 
                                  (df_filtered['method'] == method)]
        
        config_data = []
        for j, (error_col, label) in enumerate(zip(error_columns, error_labels)):
            # Extract error values for each timestep
            error_values = []
            for n in timesteps:
                subset = data_subset[data_subset['n_timesteps'] == n]
                if not subset.empty:
                    val = subset[error_col].iloc[0]
                    # Handle inf values by replacing with NaN
                    if np.isinf(val):
                        val = np.nan
                    error_values.append(val)
                else:
                    error_values.append(np.nan)
            
            config_data.append((error_col, label, error_values))
            # Collect valid values for y-axis scaling
            valid_values = [v for v in error_values if not np.isnan(v) and v > 0]
            all_error_values.extend(valid_values)
        
        plot_data.append((odeint_type, method, title, config_data))

    # Determine global y-axis limits
    if all_error_values:
        y_min = min(all_error_values) * 0.5
        y_max = max(all_error_values) * 2.0
    else:
        y_min, y_max = 1e-6, 1e0

    # Plot each configuration with shared y-axis
    for i, (odeint_type, method, title, config_data) in enumerate(plot_data):
        ax = axes[i]
        
        # Plot each error metric
        for j, (error_col, label, error_values) in enumerate(config_data):
            marker_style, color = line_styles[j]
            
            # Only plot if we have valid (non-NaN) data
            if not all(np.isnan(error_values)):
                ax.plot(timesteps, error_values, marker_style, color=color, 
                       label=label, markersize=4, linewidth=1.5)
        
        ax.set_xlabel('N_timesteps')
        ax.set_ylabel('Error Mean')
        ax.set_title(title)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)  # Set shared y-axis limits
        ax.grid(True, alpha=0.3)

    # Create shared legend in the rightmost subplot
    legend_ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center', fontsize=10)

    # Generate filename based on configuration
    filename_parts = [f'{args.experiment}_roundoff_comparison', args.precision]
    if args.scaler is not None:
        filename_parts.append(args.scaler)
    filename = '_'.join(filename_parts) + '.png'

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary for solution errors only
    print(f"\nSummary of solution error means for {args.experiment} ({args.precision}):")
    for odeint_type in ['rampde', 'torchdiffeq']:
        for method in ['rk4', 'euler']:
            subset = df_filtered[(df_filtered['odeint_type'] == odeint_type) & 
                                (df_filtered['method'] == method)]
            if not subset.empty:
                errors = [subset[subset['n_timesteps'] == n]['sol_error_mean'].iloc[0] 
                         for n in timesteps if not subset[subset['n_timesteps'] == n].empty]
                print(f"{odeint_type} {method}: {errors}")
    
    print(f"\nPlot saved as: {filename}")

if __name__ == '__main__':
    main()