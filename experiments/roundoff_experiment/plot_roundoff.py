"""
Plot roundoff error results from experiments.

This script creates publication-quality plots showing how relative errors
vary with step size h for different precision modes and scalers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from matplotlib.patches import Patch


def setup_plot_style():
    """Setup matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'lines.linewidth': 2,
        'lines.markersize': 8,
    })


def load_results(results_dir):
    """Load all experiment results."""
    results = {}
    
    for exp in ['cnf', 'otflow', 'stl10']:
        csv_file = os.path.join(results_dir, f'{exp}_roundoff_results.csv')
        if os.path.exists(csv_file):
            results[exp] = pd.read_csv(csv_file)
            print(f"Loaded {len(results[exp])} results for {exp}")
        else:
            print(f"Warning: No results found for {exp} at {csv_file}")
    
    return results


def plot_error_vs_timesteps(df, experiment_name, output_dir):
    """Plot error vs number of timesteps for a single experiment."""
    
    # Group by configuration
    configs = []
    for _, row in df.iterrows():
        if pd.isna(row.get('error', None)):  # Skip failed runs
            config = f"{row['odeint_type']}-{row['scaler_type'] or 'none'}"
            configs.append(config)
    
    unique_configs = sorted(set(configs))
    
    # Create separate plots for each method and precision
    for method in df['method'].unique():
        for precision in df['precision'].unique():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Filter data
            mask = (df['method'] == method) & (df['precision'] == precision)
            data = df[mask]
            
            # Plot solution error (left)
            for config in unique_configs:
                config_parts = config.split('-')
                odeint_type = config_parts[0]
                scaler_type = '-'.join(config_parts[1:]) if len(config_parts) > 1 else 'none'
                
                config_mask = (data['odeint_type'] == odeint_type)
                if scaler_type != 'none':
                    config_mask &= (data['scaler_type'] == scaler_type)
                else:
                    config_mask &= data['scaler_type'].isna()
                config_data = data[config_mask]
                
                if len(config_data) > 0:
                    # Group by n_timesteps and aggregate
                    grouped = config_data.groupby('n_timesteps').agg({
                        'sol_error_mean': 'mean',
                        'sol_error_std': 'mean',
                        'grad_y0_error_mean': 'mean',
                        'grad_y0_error_std': 'mean'
                    }).reset_index()
                    
                    # Solution error
                    ax1.loglog(grouped['n_timesteps'], grouped['sol_error_mean'], 
                              'o-', label=config, markersize=8)
                    
                    # Add error bars if non-deterministic
                    if grouped['sol_error_std'].max() > 0:
                        ax1.errorbar(grouped['n_timesteps'], grouped['sol_error_mean'],
                                   yerr=grouped['sol_error_std'],
                                   fmt='none', alpha=0.3)
                    
                    # Gradient error
                    if 'grad_y0_error_mean' in grouped.columns:
                        ax2.loglog(grouped['n_timesteps'], grouped['grad_y0_error_mean'],
                                  'o-', label=config, markersize=8)
                        
                        if grouped['grad_y0_error_std'].max() > 0:
                            ax2.errorbar(grouped['n_timesteps'], grouped['grad_y0_error_mean'],
                                       yerr=grouped['grad_y0_error_std'],
                                       fmt='none', alpha=0.3)
            
            # Add reference lines
            n_vals = np.array(sorted(df['n_timesteps'].unique()), dtype=float)
            
            # Add n^(-p) reference lines for RK4 (p=4) and Euler (p=1)
            p = 4 if method == 'rk4' else 1
            ref_line = n_vals**(-p) * (n_vals[0]**p * 1e-2)
            ax1.loglog(n_vals, ref_line, 'k--', alpha=0.5, label=f'$n^{{-{p}}}$')
            ax2.loglog(n_vals, ref_line, 'k--', alpha=0.5, label=f'$n^{{-{p}}}$')
            
            # Formatting
            ax1.set_xlabel('Number of timesteps')
            ax1.set_ylabel('Relative solution error')
            ax1.set_title(f'{experiment_name.upper()} - Solution Error\n{method.upper()}, {precision}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_xlabel('Number of timesteps')
            ax2.set_ylabel('Relative gradient error')
            ax2.set_title(f'{experiment_name.upper()} - Gradient Error\n{method.upper()}, {precision}')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save figure
            filename = f'error_vs_timesteps_{experiment_name}_{method}_{precision}.pdf'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()


def plot_scaler_comparison(results, output_dir):
    """Create comparison plot across all experiments and scalers."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Colors for different configurations
    colors = {
        'torchdiffeq-grad': 'blue',
        'torchmpnode-dynamic': 'green',
        'torchmpnode-grad': 'orange',
        'torchmpnode-none': 'red'
    }
    
    # Markers for different precisions
    markers = {
        'float16': 'o',
        'bfloat16': 's'
    }
    
    plot_idx = 0
    for exp_name, df in results.items():
        for method in ['euler', 'rk4']:
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Filter for this method
            method_data = df[df['method'] == method]
            
            # Plot each configuration
            for config, color in colors.items():
                config_parts = config.split('-')
                odeint_type = config_parts[0]
                scaler_type = config_parts[1] if len(config_parts) > 1 else None
                
                for precision, marker in markers.items():
                    # Filter data
                    mask = (method_data['odeint_type'] == odeint_type)
                    mask &= (method_data['precision'] == precision)
                    
                    if scaler_type:
                        mask &= (method_data['scaler_type'] == scaler_type)
                    else:
                        mask &= method_data['scaler_type'].isna()
                    
                    config_data = method_data[mask]
                    
                    if len(config_data) > 0:
                        # Group by n_timesteps
                        grouped = config_data.groupby('n_timesteps')['sol_error_mean'].mean().reset_index()
                        
                        label = f"{config}-{precision}" if plot_idx == 0 else None
                        ax.loglog(grouped['n_timesteps'], grouped['sol_error_mean'],
                                 color=color, marker=marker, markersize=6,
                                 alpha=0.8, label=label)
            
            # Formatting
            ax.set_xlabel('Number of timesteps')
            ax.set_ylabel('Relative error')
            ax.set_title(f'{exp_name.upper()} - {method.upper()}')
            ax.grid(True, alpha=0.3)
            
            # Add reference line
            n_vals = np.array(sorted(df['n_timesteps'].unique()), dtype=float)
            p = 4 if method == 'rk4' else 1
            ref_line = n_vals**(-p) * (n_vals[0]**p * 1e-2)
            ax.loglog(n_vals, ref_line, 'k--', alpha=0.3, linewidth=1)
            
            plot_idx += 1
    
    # Add legend to first subplot
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'scaler_comparison_all.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_summary_statistics(results, output_dir):
    """Generate summary statistics table."""
    
    summary = []
    
    for exp_name, df in results.items():
        # Find largest n_timesteps where each configuration breaks down
        for config in df[['odeint_type', 'scaler_type', 'precision', 'method']].drop_duplicates().values:
            odeint, scaler, precision, method = config
            
            mask = (df['odeint_type'] == odeint) & (df['precision'] == precision) & (df['method'] == method)
            if pd.notna(scaler):
                mask &= (df['scaler_type'] == scaler)
            else:
                mask &= df['scaler_type'].isna()
            
            config_data = df[mask].sort_values('n_timesteps', ascending=False)
            
            if len(config_data) > 0:
                # Find where error exceeds threshold (e.g., 10%)
                threshold = 0.1
                breakdown_n = None
                
                for _, row in config_data.iterrows():
                    if row['sol_error_mean'] > threshold:
                        breakdown_n = row['n_timesteps']
                        break
                
                # Get error at largest n_timesteps
                largest_n_error = config_data.iloc[0]['sol_error_mean']
                largest_n = config_data.iloc[0]['n_timesteps']
                
                summary.append({
                    'experiment': exp_name,
                    'method': method,
                    'precision': precision,
                    'odeint': odeint,
                    'scaler': scaler or 'none',
                    'breakdown_n': breakdown_n or 'stable',
                    f'error_at_n={largest_n}': f"{largest_n_error:.2e}"
                })
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to: {summary_file}")
    
    # Also save as formatted text
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write("Roundoff Experiment Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for exp in results.keys():
            f.write(f"\n{exp.upper()} Experiment:\n")
            f.write("-" * 40 + "\n")
            
            exp_data = summary_df[summary_df['experiment'] == exp]
            
            # Best configurations (lowest error at largest n)
            n_cols = [col for col in exp_data.columns if col.startswith('error_at_n=')]
            if n_cols:
                error_col = n_cols[0]
                best_configs = exp_data.copy()
                best_configs['error_val'] = best_configs[error_col].apply(lambda x: float(x))
                best_configs = best_configs.nsmallest(3, 'error_val')
                
                f.write(f"\nBest configurations at {error_col}:\n")
                for _, row in best_configs.iterrows():
                    f.write(f"  {row['method']}-{row['precision']}-{row['odeint']}-{row['scaler']}: "
                           f"{row[error_col]}\n")


def main():
    parser = argparse.ArgumentParser(description='Plot roundoff experiment results')
    parser.add_argument('--results_dir', type=str, 
                       default='results',
                       help='Directory containing result CSV files')
    parser.add_argument('--output_dir', type=str,
                       default='results/plots',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Make paths absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup plotting style
    setup_plot_style()
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print("No results found to plot!")
        return
    
    # Generate plots
    print("\nGenerating individual error plots...")
    for exp_name, df in results.items():
        plot_error_vs_timesteps(df, exp_name, output_dir)
    
    print("\nGenerating comparison plots...")
    plot_scaler_comparison(results, output_dir)
    
    print("\nGenerating summary statistics...")
    generate_summary_statistics(results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()