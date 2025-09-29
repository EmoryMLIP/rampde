#!/usr/bin/env python3
"""
Plot roundoff experiment results showing only rampde with specified precision and scaling
with empty torchdiffeq panels as requested. Now supports multiple experiment types.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

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
    parser = argparse.ArgumentParser(description='Plot rampde roundoff experiment results')
    parser.add_argument('--experiment', type=str, default='cnf', 
                       choices=['cnf', 'otflow'],
                       help='Experiment type (default: cnf)')
    parser.add_argument('--precision', type=str, default='float16',
                       choices=['float16', 'bfloat16'],
                       help='Precision to filter for (default: float16)')
    parser.add_argument('--scaler', type=str, default='dynamic',
                       choices=['none', 'grad', 'dynamic'],
                       help='Scaler type to filter for (default: dynamic)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result CSV files (default: results)')
    args = parser.parse_args()

    # Read the CSV file
    csv_file = os.path.join(args.results_dir, f'{args.experiment}_roundoff_results.csv')
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)

    # Filter for rampde + specified precision + specified scaler only
    # Handle NaN values in scaler_type for some experiments (like bfloat16)
    if args.scaler == 'none':
        scaler_filter = (df['scaler_type'].isna()) | (df['scaler_type'] == 'none')
    else:
        scaler_filter = (df['scaler_type'] == args.scaler)
    
    df_filtered = df[
        (df['precision'] == args.precision) & 
        (df['odeint_type'] == 'rampde') & 
        scaler_filter
    ].copy()
    
    if df_filtered.empty:
        print(f"No data found for rampde + {args.precision} + {args.scaler}")
        return

    print(f"Plotting rampde + {args.precision} + {args.scaler} configuration")
    print(f"Found {len(df_filtered)} rows of data")

    # Get error columns and labels for this experiment type
    error_config = get_error_columns(args.experiment)
    error_columns = error_config['columns']
    error_labels = error_config['labels']

    # Define line styles for distinction
    line_styles = [
        ('o-', 'blue'),      # Solution
        ('s-', 'red'),       # Grad y0
        ('^-', 'green'),     # Phi A
        ('v-', 'orange'),    # Phi c weight
        ('d-', 'purple'),    # Phi c bias
        ('p-', 'brown'),     # Phi w weight
        ('*-', 'pink'),      # Phi N Layer0 weight
        ('h-', 'gray')       # Phi N Layer0 bias
    ]

    # Create figure with 2x2 subplots + legend subplot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.3])

    axes = [
        fig.add_subplot(gs[0, 0]),  # rampde RK4
        fig.add_subplot(gs[0, 1]),  # torchdiffeq RK4 (empty)
        fig.add_subplot(gs[1, 0]),  # rampde Euler
        fig.add_subplot(gs[1, 1])   # torchdiffeq Euler (empty)
    ]

    legend_ax = fig.add_subplot(gs[:, 2])

    # Set title based on configuration
    fig.suptitle(f'{args.experiment.upper()} Roundoff Error Analysis: {args.precision} Precision - rampde + {args.scaler} scaling', fontsize=16)

    # Extract unique timestep values for x-axis
    timesteps = sorted(df_filtered['n_timesteps'].unique())

    # Plot configurations - only rampde data
    plot_configs = [
        ('rampde', 'rk4', 'rampde - RK4'),
        ('empty', 'rk4', 'torchdiffeq - RK4 (not applicable)'),
        ('rampde', 'euler', 'rampde - Euler'),
        ('empty', 'euler', 'torchdiffeq - Euler (not applicable)')
    ]

    # Collect all error values to determine global y-axis limits
    all_error_values = []
    plot_data = []

    for i, (odeint_type, method, title) in enumerate(plot_configs):
        if odeint_type == 'empty':
            # Empty plot
            config_data = []
        else:
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

    # Plot each configuration
    for i, (odeint_type, method, title, config_data) in enumerate(plot_data):
        ax = axes[i]
        
        if odeint_type == 'empty':
            # Empty plot with explanatory text
            if args.scaler == 'dynamic':
                explanation = 'Not applicable:\ntorchdiffeq does not have\ndynamic scaling'
            else:
                explanation = f'Not applicable:\nShowing only rampde\nwith {args.scaler} scaler'
            ax.text(0.5, 0.5, explanation, 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, style='italic', color='gray')
            ax.set_xlim(timesteps[0], timesteps[-1])
            ax.set_ylim(y_min, y_max)
        else:
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
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)

    # Create shared legend in the rightmost subplot
    legend_ax.axis('off')
    # Get legend from first non-empty plot
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='center', fontsize=10)

    # Generate filename based on configuration
    filename = f'{args.experiment}_roundoff_comparison_{args.precision}_rampde_{args.scaler}.png'

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary for solution errors only
    print(f"\nSummary of solution error means for rampde + {args.precision} + {args.scaler}:")
    for method in ['rk4', 'euler']:
        subset = df_filtered[df_filtered['method'] == method]
        if not subset.empty:
            errors = [subset[subset['n_timesteps'] == n]['sol_error_mean'].iloc[0] 
                     for n in timesteps if not subset[subset['n_timesteps'] == n].empty]
            print(f"rampde {method}: {errors}")
    
    print(f"\nPlot saved as: {filename}")

    # Additional summary statistics
    print(f"\nAll available combinations in the dataset:")
    available_combos = df.groupby(['precision', 'odeint_type', 'scaler_type']).size().reset_index(name='count')
    for _, row in available_combos.iterrows():
        print(f"  {row['precision']} + {row['odeint_type']} + {row['scaler_type']}: {row['count']} rows")
    
    print(f"\nAvailable timesteps: {sorted(df['n_timesteps'].unique())}")
    print(f"Available methods: {sorted(df['method'].unique())}")

if __name__ == '__main__':
    main()