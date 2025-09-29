#!/usr/bin/env python3
"""
Visualization Dashboard for rampde experiments.

This script creates comparison plots and performance visualizations from collected experiment results.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_experiment_data(csv_path):
    """Load experiment data from collect_results.py output."""
    df = pd.read_csv(csv_path)
    return df

def create_precision_comparison(df, experiment_type, metric='final_val_acc', output_dir='.'):
    """Create a bar plot comparing different precisions for a given experiment type."""
    # Filter for specific experiment type
    exp_df = df[(df['experiment_type'] == experiment_type) & (df['has_metrics'] == True)]
    
    if len(exp_df) == 0:
        print(f"No data found for experiment type: {experiment_type}")
        return
    
    # Group by precision and odeint
    if 'precision' in exp_df.columns and 'odeint' in exp_df.columns and metric in exp_df.columns:
        grouped = exp_df.groupby(['precision', 'odeint'])[metric].agg(['mean', 'std', 'count'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        precisions = grouped.index.get_level_values('precision').unique()
        odeints = grouped.index.get_level_values('odeint').unique()
        
        x = np.arange(len(precisions))
        width = 0.35
        
        for i, odeint in enumerate(odeints):
            means = []
            stds = []
            for precision in precisions:
                if (precision, odeint) in grouped.index:
                    means.append(grouped.loc[(precision, odeint), 'mean'])
                    stds.append(grouped.loc[(precision, odeint), 'std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = width * (i - len(odeints)/2 + 0.5)
            ax.bar(x + offset, means, width, yerr=stds, label=odeint, alpha=0.8)
        
        ax.set_xlabel('Precision')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{experiment_type.upper()} - {metric.replace("_", " ").title()} by Precision')
        ax.set_xticks(x)
        ax.set_xticklabels(precisions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{experiment_type}_precision_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved precision comparison plot to: {output_path}")

def create_timing_comparison(df, output_dir='.'):
    """Create scatter plots comparing forward vs backward pass times."""
    df_timing = df[(df['has_metrics'] == True) & 
                   ('avg_time_fwd' in df.columns) & 
                   ('avg_time_bwd' in df.columns)]
    
    if len(df_timing) == 0:
        print("No timing data available")
        return
    
    # Create figure with subplots for each experiment type
    exp_types = df_timing['experiment_type'].unique()
    n_types = len(exp_types)
    
    fig = plt.figure(figsize=(15, 5 * ((n_types + 2) // 3)))
    gs = gridspec.GridSpec((n_types + 2) // 3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, exp_type in enumerate(exp_types):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        exp_df = df_timing[df_timing['experiment_type'] == exp_type]
        
        # Color by precision if available
        if 'precision' in exp_df.columns:
            precisions = exp_df['precision'].unique()
            colors = plt.cm.rainbow(np.linspace(0, 1, len(precisions)))
            
            for precision, color in zip(precisions, colors):
                precision_df = exp_df[exp_df['precision'] == precision]
                ax.scatter(precision_df['avg_time_fwd'], precision_df['avg_time_bwd'], 
                          label=precision, alpha=0.6, color=color, s=100)
        else:
            ax.scatter(exp_df['avg_time_fwd'], exp_df['avg_time_bwd'], alpha=0.6, s=100)
        
        # Add diagonal line
        max_time = max(exp_df['avg_time_fwd'].max(), exp_df['avg_time_bwd'].max())
        ax.plot([0, max_time], [0, max_time], 'k--', alpha=0.3, label='y=x')
        
        ax.set_xlabel('Forward Pass Time (s)')
        ax.set_ylabel('Backward Pass Time (s)')
        ax.set_title(f'{exp_type.upper()} - Forward vs Backward Timing')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'timing_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved timing comparison plot to: {output_path}")

def create_memory_heatmap(df, output_dir='.'):
    """Create a heatmap showing memory usage across experiments."""
    if 'peak_memory_mb' not in df.columns:
        print("No memory data available")
        return
    
    df_mem = df[(df['has_metrics'] == True) & df['peak_memory_mb'].notna()]
    
    if len(df_mem) == 0:
        print("No memory data available")
        return
    
    # Pivot data for heatmap
    if 'precision' in df_mem.columns and 'odeint' in df_mem.columns:
        pivot_data = df_mem.pivot_table(
            values='peak_memory_mb',
            index='experiment_type',
            columns=['precision', 'odeint'],
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Peak Memory (MB)'}, ax=ax)
        
        ax.set_title('Peak Memory Usage Heatmap')
        ax.set_xlabel('Precision / ODE Integrator')
        ax.set_ylabel('Experiment Type')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'memory_heatmap.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved memory heatmap to: {output_path}")

def create_speedup_plots(df, baseline_precision='float32', output_dir='.'):
    """Create speedup plots comparing different precisions to baseline."""
    df_speed = df[(df['has_metrics'] == True) & 
                  ('avg_time_fwd' in df.columns) & 
                  ('avg_time_bwd' in df.columns)]
    
    if len(df_speed) == 0:
        print("No timing data for speedup calculation")
        return
    
    # Calculate total time
    df_speed['total_time'] = df_speed['avg_time_fwd'] + df_speed['avg_time_bwd']
    
    # Group by experiment type and precision
    results = []
    for exp_type in df_speed['experiment_type'].unique():
        exp_df = df_speed[df_speed['experiment_type'] == exp_type]
        
        # Get baseline time
        baseline_df = exp_df[exp_df['precision'] == baseline_precision]
        if len(baseline_df) == 0:
            continue
            
        baseline_time = baseline_df['total_time'].mean()
        
        # Calculate speedup for each precision
        for precision in exp_df['precision'].unique():
            precision_df = exp_df[exp_df['precision'] == precision]
            avg_time = precision_df['total_time'].mean()
            speedup = baseline_time / avg_time if avg_time > 0 else 0
            
            results.append({
                'experiment_type': exp_type,
                'precision': precision,
                'speedup': speedup,
                'count': len(precision_df)
            })
    
    speedup_df = pd.DataFrame(results)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot for grouped bar plot
    pivot_speedup = speedup_df.pivot(index='experiment_type', columns='precision', values='speedup')
    pivot_speedup.plot(kind='bar', ax=ax)
    
    # Add horizontal line at speedup=1
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Experiment Type')
    ax.set_ylabel(f'Speedup vs {baseline_precision}')
    ax.set_title(f'Performance Speedup Relative to {baseline_precision}')
    ax.legend(title='Precision', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'speedup_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved speedup comparison to: {output_path}")

def create_convergence_plots(df, experiment_type, n_examples=3, output_dir='.'):
    """Create convergence plots showing training progress for selected experiments."""
    exp_df = df[(df['experiment_type'] == experiment_type) & (df['has_metrics'] == True)]
    
    if len(exp_df) == 0:
        print(f"No data found for experiment type: {experiment_type}")
        return
    
    # Select a few representative experiments
    selected = exp_df.sample(n=min(n_examples, len(exp_df)))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (_, exp) in enumerate(selected.iterrows()):
        # Read the full metrics CSV
        csv_path = os.path.join(exp['result_dir'], f"{exp['folder_name']}.csv")
        if not os.path.exists(csv_path):
            continue
            
        try:
            metrics_df = pd.read_csv(csv_path)
            
            # Rename columns if needed
            if 'step' in metrics_df.columns:
                metrics_df = metrics_df.rename(columns={'step': 'iter'})
            if 'iteration' in metrics_df.columns:
                metrics_df = metrics_df.rename(columns={'iteration': 'iter'})
            
            # Plot different metrics
            if idx < 4:  # We have 4 subplots
                ax = axes[idx]
                
                # Plot losses if available
                if 'train_loss' in metrics_df.columns:
                    ax.plot(metrics_df['iter'], metrics_df['train_loss'], 
                           label='Train Loss', alpha=0.8)
                if 'val_loss' in metrics_df.columns:
                    ax.plot(metrics_df['iter'], metrics_df['val_loss'], 
                           label='Val Loss', alpha=0.8)
                elif 'running_loss' in metrics_df.columns:
                    ax.plot(metrics_df['iter'], metrics_df['running_loss'], 
                           label='Running Loss', alpha=0.8)
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.set_title(f"{exp['precision']} - {exp['odeint']}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            print(f"Could not read metrics for {exp['folder_name']}: {e}")
    
    plt.suptitle(f'{experiment_type.upper()} - Training Convergence Examples')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_type}_convergence.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plots to: {output_path}")

def create_summary_dashboard(df, output_dir='.'):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Experiment count by type
    ax1 = fig.add_subplot(gs[0, 0])
    df['experiment_type'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Experiments by Type')
    ax1.set_xlabel('Experiment Type')
    ax1.set_ylabel('Count')
    
    # 2. Completion status
    ax2 = fig.add_subplot(gs[0, 1])
    status_counts = pd.Series({
        'Completed': df['has_final_model'].sum(),
        'Has Metrics': df['has_metrics'].sum() - df['has_final_model'].sum(),
        'No Metrics': (~df['has_metrics']).sum(),
        'Emergency Stop': df['has_emergency_stop'].sum()
    })
    status_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Experiment Status')
    
    # 3. Precision distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'precision' in df.columns:
        df['precision'].value_counts().plot(kind='bar', ax=ax3)
        ax3.set_title('Experiments by Precision')
        ax3.set_xlabel('Precision')
        ax3.set_ylabel('Count')
    
    # 4. Average forward/backward time ratio
    ax4 = fig.add_subplot(gs[1, :2])
    if 'avg_time_fwd' in df.columns and 'avg_time_bwd' in df.columns:
        df_time = df[df['avg_time_fwd'].notna() & df['avg_time_bwd'].notna()]
        df_time['bwd_fwd_ratio'] = df_time['avg_time_bwd'] / df_time['avg_time_fwd']
        df_time.boxplot(column='bwd_fwd_ratio', by='experiment_type', ax=ax4)
        ax4.set_title('Backward/Forward Time Ratio by Experiment')
        ax4.set_xlabel('Experiment Type')
        ax4.set_ylabel('Backward/Forward Time Ratio')
    
    # 5. Memory usage distribution
    ax5 = fig.add_subplot(gs[1, 2])
    if 'peak_memory_mb' in df.columns:
        df_mem = df[df['peak_memory_mb'].notna()]
        df_mem.boxplot(column='peak_memory_mb', by='experiment_type', ax=ax5)
        ax5.set_title('Memory Usage by Experiment')
        ax5.set_xlabel('Experiment Type')
        ax5.set_ylabel('Peak Memory (MB)')
    
    # 6. Best performers table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create best performers table
    best_performers = []
    for exp_type in df['experiment_type'].unique():
        exp_df = df[df['experiment_type'] == exp_type]
        if 'final_val_acc' in exp_df.columns and exp_df['final_val_acc'].notna().any():
            best_idx = exp_df['final_val_acc'].idxmax()
            best = exp_df.loc[best_idx]
            best_performers.append({
                'Experiment': exp_type,
                'Best Val Acc': f"{best['final_val_acc']:.4f}" if pd.notna(best['final_val_acc']) else 'N/A',
                'Precision': best.get('precision', 'N/A'),
                'Method': best.get('method', 'N/A'),
                'Time (s)': f"{best.get('avg_time_fwd', 0) + best.get('avg_time_bwd', 0):.2f}"
            })
    
    if best_performers:
        table_df = pd.DataFrame(best_performers)
        table = ax6.table(cellText=table_df.values,
                         colLabels=table_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Best Performers by Experiment Type', pad=20)
    
    plt.suptitle('rampde Experiment Summary Dashboard', fontsize=16, y=0.98)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'summary_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary dashboard to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize rampde experiment results')
    parser.add_argument('input', type=str, help='Input CSV file from collect_results.py')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for plots')
    parser.add_argument('--experiment', type=str,
                       help='Specific experiment type to visualize')
    parser.add_argument('--plots', type=str, nargs='+', 
                       choices=['precision', 'timing', 'memory', 'speedup', 'convergence', 'dashboard', 'all'],
                       default=['all'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading experiment data from: {args.input}")
    df = load_experiment_data(args.input)
    print(f"Loaded {len(df)} experiments")
    
    # Generate requested plots
    if 'all' in args.plots or 'dashboard' in args.plots:
        create_summary_dashboard(df, args.output_dir)
    
    if 'all' in args.plots or 'precision' in args.plots:
        if args.experiment:
            create_precision_comparison(df, args.experiment, output_dir=args.output_dir)
        else:
            for exp_type in df['experiment_type'].unique():
                create_precision_comparison(df, exp_type, output_dir=args.output_dir)
    
    if 'all' in args.plots or 'timing' in args.plots:
        create_timing_comparison(df, args.output_dir)
    
    if 'all' in args.plots or 'memory' in args.plots:
        create_memory_heatmap(df, args.output_dir)
    
    if 'all' in args.plots or 'speedup' in args.plots:
        create_speedup_plots(df, output_dir=args.output_dir)
    
    if 'all' in args.plots or 'convergence' in args.plots:
        if args.experiment:
            create_convergence_plots(df, args.experiment, output_dir=args.output_dir)
        else:
            for exp_type in df['experiment_type'].unique()[:2]:  # Just do first 2 to avoid too many plots
                create_convergence_plots(df, exp_type, output_dir=args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")

if __name__ == '__main__':
    main()