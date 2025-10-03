#!/usr/bin/env python3
"""
STL10 Convergence Plot Generator - TikZ Version

Creates publication-quality TikZ convergence plots with systematic
color, marker, and line style design for maximum clarity.

Design Logic:
- Colors: Colorblind-friendly palette (blue, orange, purple)  
- Markers: Different shapes for solver types (circle=TD, square=MP)
- Line styles: Different patterns for scaler types or solver distinction
- Excludes float32 (keeps only tfloat32 for float32-like precision)

Now reads directly from raw experiment directories for transparency.
"""

import pandas as pd
from pathlib import Path
import argparse
import fnmatch


def load_raw_data_from_directories(raw_data_dir: Path, dir_pattern: str = '*', metric: str = 'train_loss'):
    """Load data directly from raw experiment directories."""
    
    # Find matching directories using fnmatch for complex patterns
    all_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir() and d.name.startswith('stl10_')]
    
    if dir_pattern == '*':
        matching_dirs = all_dirs
    else:
        # Apply the filter pattern
        full_pattern = f"stl10*{dir_pattern}"
        matching_dirs = [d for d in all_dirs if fnmatch.fnmatch(d.name, full_pattern)]
    
    if not matching_dirs:
        raise FileNotFoundError(f"No directories found matching pattern: stl10*{dir_pattern}")
    
    print(f"Found {len(matching_dirs)} directories matching 'stl10*{dir_pattern}'")
    
    all_data = []
    directory_info = []
    
    for exp_dir in matching_dirs:
        print(f"Processing: {exp_dir.name}")
        
        # Parse directory name to extract configuration
        dir_name = exp_dir.name
        
        # Read args.csv to get configuration details
        args_file = exp_dir / 'args.csv'
        if not args_file.exists():
            print(f"  Warning: No args.csv found in {dir_name}, skipping")
            continue
            
        args_df = pd.read_csv(args_file)
        if len(args_df) == 0:
            print(f"  Warning: Empty args.csv in {dir_name}, skipping")
            continue
            
        config_row = args_df.iloc[0]
        precision = config_row['precision_str']
        solver = config_row['odeint_type']
        width = config_row['width']
        seed = config_row['seed']
        
        # Read training data CSV
        data_csv = exp_dir / f"{dir_name}.csv"
        if not data_csv.exists():
            print(f"  Warning: No training data CSV found in {dir_name}, skipping")
            continue
            
        train_df = pd.read_csv(data_csv)
        if len(train_df) == 0:
            print(f"  Warning: Empty training data in {dir_name}, skipping")
            continue
        
        # Convert time from seconds to hours (total time = forward + backward)
        if 'time_fwd_sum' in train_df.columns and 'time_bwd_sum' in train_df.columns:
            train_df['time_hours'] = (train_df['time_fwd_sum'] + train_df['time_bwd_sum']) / 3600
        elif 'time_fwd_sum' in train_df.columns:
            print(f"  Warning: No time_bwd_sum column in {dir_name}, using only forward time")
            train_df['time_hours'] = train_df['time_fwd_sum'] / 3600
        else:
            print(f"  Warning: No time columns found in {dir_name}, skipping")
            continue
            
        # Extract the requested metric
        if metric not in train_df.columns:
            print(f"  Warning: Metric '{metric}' not found in {dir_name}, skipping")
            continue
            
        # Create configuration string
        if precision == 'float16':
            # Determine scaler type from directory name
            if 'dynamic' in dir_name:
                scaler = 'Dyn'
            elif 'grad' in dir_name:
                scaler = 'Grad'
            elif 'none' in dir_name:
                scaler = 'None'
            else:
                scaler = 'Unknown'
            
            solver_short = 'MP' if solver == 'rampde' else 'TD'
            config = f"{precision}+{solver_short}+{scaler}"
        else:
            solver_short = 'MP' if solver == 'rampde' else 'TD'
            config = f"{precision}+{solver_short}"
        
        # Skip float32 configurations and problematic float16+TD configurations
        if config.startswith('float32+'):
            print(f"  Skipping float32 config: {config}")
            continue
        
        # Skip problematic float16+TD configurations that failed/have minimal data
        if config in ['float16+TD+Grad', 'float16+TD+None', 'float16+MP+Grad']:
            print(f"  Skipping problematic config: {config}")
            continue
            
        # Add configuration and directory info to each row
        train_subset = train_df[['time_hours', metric]].copy()
        train_subset['config'] = config
        train_subset['directory'] = dir_name
        train_subset['precision'] = precision
        train_subset['solver'] = solver
        train_subset['width'] = width
        train_subset['seed'] = seed
        
        all_data.append(train_subset)
        
        # Store directory info
        final_val_acc = train_df['val_acc'].iloc[-1] if 'val_acc' in train_df.columns else 0.0
        max_memory = train_df['max_memory_mb'].max() if 'max_memory_mb' in train_df.columns else 0.0
        
        directory_info.append({
            'config': config,
            'directory': dir_name,
            'precision': precision,
            'solver': solver,
            'final_val_acc': final_val_acc,
            'max_memory_mb': max_memory,
            'width': width,
            'seed': seed
        })
        
        print(f"  ✓ Loaded {config} (Val Acc: {final_val_acc:.3f}, Memory: {max_memory:.1f} MB)")
    
    if not all_data:
        raise ValueError("No valid data found in any directory")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    dir_info_df = pd.DataFrame(directory_info)
    
    return combined_df, dir_info_df


def create_design_csvs_and_tikz(combined_df: pd.DataFrame, dir_info_df: pd.DataFrame, output_dir: Path, metric: str = 'train_loss'):
    """Create individual CSV files for each configuration with accessible design."""
    
    # Get unique configurations (already filtered to exclude float32)
    configs = combined_df['config'].unique()
    
    print(f"Using {len(configs)} configurations (excluded float32):")
    for c in configs:
        print(f"  - {c}")
    
    # ACCESSIBLE DESIGN SYSTEM
    # Color palette: Match the working file exactly
    precision_colors = {
        'tfloat32': 'rgb,255:red,148;green,103;blue,189',  # Purple
        'bfloat16': 'rgb,255:red,31;green,119;blue,180',   # Blue
        'float16': 'rgb,255:red,255;green,127;blue,14',    # Orange (for Dyn)
        'float16_none': 'rgb,255:red,44;green,160;blue,44' # Green (for None)
    }
    
    # Markers: Different shapes for different solver types (matching reference image)
    solver_markers = {
        'torchdiffeq': 'triangle*',    # Filled triangle
        'rampde': 'square*'       # Filled square
    }

    # Line styles: Simple and consistent approach
    # rampde (MP) = solid, torchdiffeq (TD) = dashed
    solver_styles = {
        'torchdiffeq': 'dashed',
        'rampde': 'solid'
    }
    
    # Create individual CSV files and collect plot commands
    plot_commands = []
    
    for config in configs:
        # Filter data for this configuration
        config_data = combined_df[combined_df['config'] == config][['time_hours', metric]].copy()
        config_data = config_data.rename(columns={metric: 'value'})
        
        # Get configuration info
        config_info = dir_info_df[dir_info_df['config'] == config].iloc[0]
        precision = config_info['precision']
        solver = config_info['solver']
        val_acc = config_info['final_val_acc']
        memory_mb = config_info['max_memory_mb']
        directory = config_info['directory']
        
        # Save individual CSV
        csv_filename = f'stl10_{metric}_{config.replace("+", "_")}.csv'
        csv_path = output_dir / csv_filename
        config_data.to_csv(csv_path, index=False)
        
        # Determine styling - match working file exactly
        if precision == 'tfloat32':
            tikz_color = f"{{{precision_colors['tfloat32']}}}"
        elif precision == 'bfloat16':
            tikz_color = f"{{{precision_colors['bfloat16']}}}"
        elif precision == 'float16' and 'None' in config:
            tikz_color = f"{{{precision_colors['float16_none']}}}"
        elif precision == 'float16':  # Dyn or other float16
            tikz_color = f"{{{precision_colors['float16']}}}"
        else:
            tikz_color = "{rgb,255:red,127;green,127;blue,127}"  # Gray fallback

        marker = solver_markers.get(solver, 'o')

        # Determine line style based on solver
        line_style = solver_styles.get(solver, 'solid')

        # Create readable legend entry
        legend_entry = config.replace('+', ' + ')
        
        # Create plot command with accessibility features
        # Create plot label for referencing
        plot_label = f"plot:{precision.lower()}-{solver.replace('torchdiffeq', 'td').replace('rampde', 'mp')}"
        if precision == 'float16' and 'Dyn' in config:
            plot_label = "plot:float16-dyn"
        elif precision == 'float16' and 'None' in config:
            plot_label = "plot:float16-none"
        elif precision == 'float16' and 'Grad' in config:
            plot_label = "plot:float16-grad"

        plot_command = f"""% {config} (precision: {precision}, solver: {solver})
% Directory: {directory}
% Final validation accuracy: {val_acc:.3f}, Memory: {memory_mb:.2f} MB
\\addplot[
    color={tikz_color},
    line width=1.2pt,
    {line_style},
    mark={{{marker}}},
    mark size=2.5pt,
    mark repeat=20,
    mark phase=10
] table [
    x=time_hours,
    y=value,
    col sep=comma
] {{{csv_filename}}};
\\label{{{plot_label}}}

"""
        plot_commands.append((plot_command, config, precision, solver, plot_label))
    
    return plot_commands, configs


def create_grouped_legend_content(plot_commands):
    """Create plot content with external matrix legend using node and ref."""
    plot_content = []
    legend_lines = []

    # Sort plot commands by solver then precision for proper grouping
    # torchdiffeq first, then rampde
    # Within each group: tfloat32, bfloat16, then float16 variants
    def sort_key(x):
        plot_cmd, config, precision, solver, plot_label = x
        solver_order = 0 if solver == 'torchdiffeq' else 1

        # Precision order: tfloat32, bfloat16, then float16 variants
        if precision == 'tfloat32':
            precision_order = 0
        elif precision == 'bfloat16':
            precision_order = 1
        elif precision == 'float16' and 'Dyn' in config:
            precision_order = 2
        elif precision == 'float16' and 'None' in config:
            precision_order = 3
        else:
            precision_order = 4

        return (solver_order, precision_order)

    sorted_commands = sorted(plot_commands, key=sort_key)

    # Add plots and collect legend entries
    current_solver = None
    torchdiffeq_refs = []
    rampde_refs = []

    for plot_cmd, config, precision, solver, plot_label in sorted_commands:
        # Add group comment when solver changes
        if solver != current_solver:
            if solver == 'torchdiffeq':
                plot_content.append("% === GROUP 1: torchdiffeq ===\n\n")
                current_solver = solver
            elif solver == 'rampde':
                plot_content.append("% === GROUP 2: rampde ===\n\n")
                current_solver = solver

        plot_content.append(plot_cmd)

        # Collect references for external legend
        if solver == 'torchdiffeq':
            if precision == 'tfloat32':
                torchdiffeq_refs.append(f"\\quad \\ref{{{plot_label}}} tfloat32")
            elif precision == 'bfloat16':
                torchdiffeq_refs.append(f"\\quad \\ref{{{plot_label}}} bfloat16")
        elif solver == 'rampde':
            if precision == 'tfloat32':
                rampde_refs.append(f"\\quad \\ref{{{plot_label}}} tfloat32")
            elif precision == 'bfloat16':
                rampde_refs.append(f"\\quad \\ref{{{plot_label}}} bfloat16")
            elif precision == 'float16' and 'Dyn' in config:
                rampde_refs.append(f"\\quad \\ref{{{plot_label}}} float16+Dyn")
            elif precision == 'float16' and 'None' in config:
                rampde_refs.append(f"\\quad \\ref{{{plot_label}}} float16+None")

    # Build external legend node
    sep = " \\\\ "
    legend_content = f"""
% Hierarchical legend using external matrix
\\node[
    anchor=north east,
    inner sep=0.4em,
    outer sep=0.1em,
    fill=white,
    fill opacity=0.8,
    text opacity=1,
    draw=gray!30,
    rounded corners=2pt,
    font=\\footnotesize,
    align=left,
] at ([yshift=-0.5ex, xshift=-0.5ex]current axis.north east) {{
    $\\blacktriangle$ \\textbf{{torchdiffeq}} (dashed): \\\\
{sep.join(torchdiffeq_refs)} \\\\[0.3em]
    $\\blacksquare$ \\textbf{{rampde}} (solid): \\\\
{sep.join(rampde_refs)} \\\\
}};"""

    return "".join(plot_content), legend_content


def create_tikz_file(raw_data_dir: Path, output_dir: Path, dir_pattern: str = '*', metric: str = 'train_loss'):
    """Create accessible TikZ file for the convergence plot."""
    
    # Load data directly from raw directories
    combined_df, dir_info_df = load_raw_data_from_directories(raw_data_dir, dir_pattern, metric)
    
    # Save the consolidated CSV file
    consolidated_csv = output_dir / f'stl10_{metric}_consolidated_data.csv'
    combined_df.to_csv(consolidated_csv, index=False)
    print(f"Saved consolidated data to: {consolidated_csv}")
    
    # Create plot commands
    plot_commands, configs = create_design_csvs_and_tikz(combined_df, dir_info_df, output_dir, metric)
    
    # Metric-specific settings
    if metric == 'train_loss':
        ylabel = 'Training Loss'
        title = 'Training Loss vs Training Time'
        use_log = True
    elif metric == 'val_loss':
        ylabel = 'Validation Loss'
        title = 'Validation Loss vs Training Time'
        use_log = True
    elif metric == 'val_acc':
        ylabel = 'Validation Accuracy'
        title = 'Validation Accuracy vs Training Time'
        use_log = False
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Axis type
    axis_type = 'semilogyaxis' if use_log else 'axis'
    
    # Use the loaded directory info for header comments (already filtered)
    dir_info_filtered = dir_info_df
    
    # Create header comments
    header_comments = [
        f"% STL10 {title} - TikZ Version",
        "% Colorblind-friendly colors, multiple visual encodings for clarity",
        "% Design: Color=precision, Marker=solver, LineStyle=scaler/solver distinction",
        "% Excluded float32 configurations (kept tfloat32 only)",
        "% Generated from: /Users/lruthot/Dropbox/Projects/Overleaf/2025-SISC-MixedPrecision/results_paper/outputs/fig_stl10_convergence/stl10_time_based_convergence.png",
        "% Data extracted from directories:"
    ]
    
    for _, row in dir_info_filtered.iterrows():
        header_comments.append(f"% - {row['config']}: {row['directory']}")
    
    # Create complete TikZ content matching reference format exactly
    plot_content, legend_content = create_grouped_legend_content(plot_commands)

    tikz_content = "\n".join(header_comments) + f"""

\\begin{{tikzpicture}}
\\begin{{{axis_type}}}[
    width=14cm,
    height=9cm,
    xlabel={{Training Time (hours)}},
    ylabel={{{ylabel}}},
    title={{{title}}},
    % legend disabled - using external matrix legend instead
    grid=major,
    grid style={{gray!20}},
    xmin=0, xmax=10.5,
    ymin=0.006, ymax=3.0,
    tick label style={{font=\\small}},
    label style={{font=\\normalsize}},
    title style={{font=\\large}},
    line width=1.2pt,
    mark size=2.5pt,
    every axis plot/.append style={{thick}}
]

{plot_content}
\\end{{{axis_type}}}
{legend_content}

\\end{{tikzpicture}}"""
    
    # Save TikZ file
    tikz_filename = f'stl10_{metric}_convergence.tex'
    tikz_path = output_dir / tikz_filename
    
    with open(tikz_path, 'w') as f:
        f.write(tikz_content)
    
    # Create standalone LaTeX file for testing
    standalone_content = f"""\\documentclass{{standalone}}
\\usepackage{{pgfplots}}
\\usepackage{{xcolor}}
\\usepackage{{amsmath,amssymb}}
\\usetikzlibrary{{matrix}}
\\pgfplotsset{{compat=1.17}}

\\begin{{document}}
{tikz_content}
\\end{{document}}"""
    
    standalone_filename = f'stl10_{metric}_standalone.tex'
    standalone_path = output_dir / standalone_filename
    
    with open(standalone_path, 'w') as f:
        f.write(standalone_content)
    
    print(f"Created TikZ file: {tikz_path}")
    print(f"Created standalone file: {standalone_path}")
    print(f"Created {len(configs)} individual CSV files for configurations:")
    for config in configs:
        print(f"  - {config}")
    
    # Print design summary
    print(f"\nDesign Summary:")
    print(f"  Colors: Blue (bfloat16), Orange (float16), Purple (tfloat32)")
    print(f"  Markers: Circle (torchdiffeq), Square (rampde)")
    print(f"  Line styles: Solid/Dashed/Dotted for distinction")
    print(f"  Excluded: float32 configurations (kept tfloat32)")
    
    return tikz_path


def main():
    parser = argparse.ArgumentParser(
        description='Create TikZ convergence plot for STL10 data by reading directly from raw experiment directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all directories
  python plot_stl10_convergence.py
  
  # Filter by seed
  python plot_stl10_convergence.py --filter "*seed42*"
  
  # Filter by width
  python plot_stl10_convergence.py --filter "*width_128*"
  
  # Multiple filters (combine patterns)
  python plot_stl10_convergence.py --filter "*seed42*width_128*"
        """
    )
    parser.add_argument('--raw-data-dir', type=str,
                       default='raw_data/ode_stl10',
                       help='Directory containing raw experiment directories')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/fig_stl10_train_loss',
                       help='Output directory for generated files')
    parser.add_argument('--filter', type=str, default='*',
                       help='Directory filter pattern (e.g., "*seed42*", "*width_128*")')
    parser.add_argument('--metric', type=str, default='train_loss',
                       choices=['train_loss', 'val_loss', 'val_acc'],
                       help='Metric to plot')
    
    args = parser.parse_args()
    
    raw_data_dir = Path(args.raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading from: {raw_data_dir}")
    print(f"Filter pattern: {args.filter}")
    print(f"Output directory: {output_dir}")
    print(f"Metric: {args.metric}")
    print()
    
    tikz_path = create_tikz_file(raw_data_dir, output_dir, args.filter, args.metric)
    print(f"\nTikZ plot ready: {tikz_path}")


if __name__ == "__main__":
    main()