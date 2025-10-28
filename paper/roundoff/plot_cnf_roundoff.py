#!/usr/bin/env python3
"""
CNF Roundoff Error Plot Generator - TikZ Version

Creates publication-quality TikZ plots for CNF roundoff error analysis.
Plots relative errors vs time steps for different configurations.

Design Logic:
- Colors: Colorblind-friendly palette (blue, orange)
- Markers: Different shapes for solver types (circle=TD, square=MP)
- Line styles: Different patterns for scaler types
"""

import pandas as pd
from pathlib import Path
import argparse


def load_cnf_roundoff_data(csv_path: Path, method: str, precision: str,
                          odeint_type: str, scaler_type: str):
    """Load and filter CNF roundoff data based on configuration."""

    df = pd.read_csv(csv_path)

    # Filter data based on arguments
    filtered_df = df[
        (df['method'] == method) &
        (df['precision'] == precision) &
        (df['odeint_type'] == odeint_type)
    ]

    # Handle scaler_type filtering - empty string or NaN means 'none'
    if scaler_type == 'none':
        filtered_df = filtered_df[
            (filtered_df['scaler_type'].isna()) |
            (filtered_df['scaler_type'] == '') |
            (filtered_df['scaler_type'] == 'none')
        ]
    else:
        filtered_df = filtered_df[filtered_df['scaler_type'] == scaler_type]

    if len(filtered_df) == 0:
        raise ValueError(f"No data found for configuration: {method}, {precision}, {odeint_type}, {scaler_type}")

    print(f"Found {len(filtered_df)} data points for configuration")

    # Get all error columns (columns ending with '_error_mean')
    error_columns = [col for col in df.columns if col.endswith('_error_mean')]

    # Prepare data for plotting
    plot_data = []

    # Sort by n_timesteps for proper line plotting
    filtered_df = filtered_df.sort_values('n_timesteps')

    for col in error_columns:
        # Extract clean column name for legend
        col_name = col.replace('grad_', '').replace('_error_mean', '')

        # Create data subset
        subset = filtered_df[['n_timesteps', col]].copy()
        subset['error_type'] = col_name
        subset = subset.rename(columns={col: 'error_value'})

        plot_data.append(subset)

    # Combine all error data
    combined_data = pd.concat(plot_data, ignore_index=True)

    return combined_data, error_columns


def create_tikz_plot(data: pd.DataFrame, error_columns: list, output_dir: Path,
                    config_str: str, show_legend: bool = True, show_xlabel: bool = True,
                    show_ylabel: bool = True):
    """Create TikZ plot and individual CSV files."""

    # Create individual CSV files for each error type
    plot_commands = []

    # Color palette for different error types
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]

    # Markers for variety
    markers = ['o', 'square*', 'triangle*', 'diamond*', 'star', 'pentagon*']

    for i, error_col in enumerate(error_columns):
        # Extract error type name
        error_type = error_col.replace('grad_', '').replace('_error_mean', '')

        # Filter data for this error type
        error_data = data[data['error_type'] == error_type][['n_timesteps', 'error_value']].copy()

        # Save individual CSV
        csv_filename = f'cnf_roundoff_{config_str}_{error_type.replace(".", "_")}.csv'
        csv_path = output_dir / csv_filename
        error_data.to_csv(csv_path, index=False)

        # Choose color and marker
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Convert hex color to RGB for TikZ
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        rgb = hex_to_rgb(color)
        tikz_color = f"{{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}"

        # Create meaningful legend entry
        legend_mapping = {
            'sol': 'Solution',
            'y0': 'Initial Value',
            'grad_y0': 'Gradient w.r.t. y₀',
            'hyper_net.fc1.weight': 'Layer 1 Weights',
            'hyper_net.fc1.bias': 'Layer 1 Bias',
            'hyper_net.fc2.weight': 'Layer 2 Weights',
            'hyper_net.fc2.bias': 'Layer 2 Bias',
            'hyper_net.fc3.weight': 'Output Weights',
            'hyper_net.fc3.bias': 'Output Bias'
        }
        legend_entry = legend_mapping.get(error_type, error_type)

        # Choose line style - dashed for solution (forward pass), solid for derivatives (backward pass)
        line_style = "dashed" if error_type == 'sol' else "solid"

        # Create plot command
        plot_command = f"""% Error type: {error_type}
\\addplot[
    color={tikz_color},
    line width=1.2pt,
    {line_style},
    mark={{{marker}}},
    mark size=2pt
] table [
    x=n_timesteps,
    y=error_value,
    col sep=comma
] {{{csv_filename}}};"""

        # Add label for hierarchical legend (don't use addlegendentry)
        if show_legend:
            plot_command += f"\n\\label{{plot:{error_type.replace('.', '-')}}}"

        plot_command += "\n\n"
        plot_commands.append(plot_command)

    # Build axis options
    axis_options = [
        "width=14cm",
        "height=10cm",
        "grid=major",
        "grid style={gray!20}",
        "xmin=50, xmax=4096",
        "ymin=1e-4, ymax=1",
        "xtick={64, 128, 256, 512, 1024, 2048, 4096}",
        "xticklabels={64, 128, 256, 512, 1024, 2048, 4096}",
        "tick label style={font=\\Large}",
        "label style={font=\\huge}",
        "line width=1.2pt",
        "mark size=2pt",
        "every axis plot/.append style={thick}"
    ]

    if show_xlabel:
        axis_options.insert(2, "xlabel={Number of Time Steps}")
    if show_ylabel:
        axis_options.insert(3, "ylabel={Relative Error}")
    # Note: legend disabled - using external hierarchical legend instead

    axis_options_str = ",\n    ".join(axis_options)

    # Create hierarchical legend if enabled
    hierarchical_legend = ""
    if show_legend:
        # Separate solution and derivative error types
        solution_entries = []
        derivative_entries = []

        for col in error_columns:
            error_type = col.replace('grad_', '').replace('_error_mean', '')
            legend_entry = legend_mapping.get(error_type, error_type)
            label_ref = error_type.replace('.', '-')

            if error_type == 'sol':
                solution_entries.append(f"\\quad \\ref{{plot:{label_ref}}} {legend_entry}")
            else:
                derivative_entries.append(f"\\quad \\ref{{plot:{label_ref}}} {legend_entry}")

        # Build hierarchical legend content
        legend_lines = []
        if solution_entries:
            legend_lines.append("\\textbf{Forward} (dashed): \\\\")
            legend_lines.extend(solution_entries)
            legend_lines.append("\\\\[0.3em]")

        if derivative_entries:
            legend_lines.append("\\textbf{Backward} (solid): \\\\")
            legend_lines.extend(derivative_entries)

        legend_content = " \\\\\n    ".join(legend_lines)
        hierarchical_legend = f"""

% Hierarchical legend using external matrix - positioned to the right outside plot
\\node[
    anchor=north west,
    inner sep=0.4em,
    outer sep=0.1em,
    fill=white,
    fill opacity=0.8,
    text opacity=1,
    draw=gray!30,
    rounded corners=2pt,
    font=\\Large,
    align=left,
] at ([xshift=1cm]current axis.north east) {{
    {legend_content} \\\\
}};"""

    # Create TikZ content
    tikz_content = f"""% CNF Roundoff Error Plot - Configuration: {config_str}
% Relative errors vs number of time steps

\\begin{{tikzpicture}}
\\begin{{loglogaxis}}[
    {axis_options_str}
]

{"".join(plot_commands)}\\end{{loglogaxis}}{hierarchical_legend}

\\end{{tikzpicture}}"""

    # Save TikZ file
    tikz_filename = f'cnf_roundoff_{config_str}.tex'
    tikz_path = output_dir / tikz_filename

    with open(tikz_path, 'w') as f:
        f.write(tikz_content)

    # Create standalone LaTeX file
    standalone_content = f"""\\documentclass{{standalone}}
\\usepackage{{pgfplots}}
\\usepackage{{xcolor}}
\\usepackage{{amsmath,amssymb}}
\\usetikzlibrary{{matrix,positioning}}
\\pgfplotsset{{compat=1.17}}

\\begin{{document}}
{tikz_content}
\\end{{document}}"""

    standalone_filename = f'cnf_roundoff_{config_str}_standalone.tex'
    standalone_path = output_dir / standalone_filename

    with open(standalone_path, 'w') as f:
        f.write(standalone_content)

    print(f"Created TikZ file: {tikz_path}")
    print(f"Created standalone file: {standalone_path}")
    print(f"Created {len(error_columns)} individual CSV files")

    return tikz_path


def create_combined_2x2_figure(output_dir: Path):
    """Create combined 2x2 figure from individual plots."""

    # Define the four configurations for the 2x2 grid
    configs = [
        ("torchdiffeq", "float16", "grad", "euler", "top-left"),
        ("rampde", "float16", "dynamic", "euler", "top-right"),
        ("torchdiffeq", "float16", "grad", "rk4", "bottom-left"),
        ("rampde", "float16", "dynamic", "rk4", "bottom-right")
    ]

    # Create combined TikZ content - exact copy of original format
    combined_content = """\\begin{tikzpicture}[node distance=0.3cm]

% (1) Top left plot - torchdiffeq Euler
\\node (euler-torch) {\\scalebox{0.5}{\\input{cnf_roundoff_torchdiffeq_float16_grad_euler.tex}}};

% (2) Top right plot - positioned relative to right border of (1)
\\node (euler-mp) [right=.3cm of euler-torch.east, anchor=west] {\\scalebox{0.5}{\\input{cnf_roundoff_rampde_float16_dynamic_euler.tex}}};

% (3) Bottom left plot - positioned relative to (1)
\\node (rk4-torch) [below=0cm of euler-torch] {\\scalebox{0.5}{\\input{cnf_roundoff_torchdiffeq_float16_grad_rk4.tex}}};

% (4) Bottom right plot - positioned relative to (3)
\\node (rk4-mp) [right=-.1cm of rk4-torch.east, anchor=west] {\\scalebox{0.5}{\\input{cnf_roundoff_rampde_float16_dynamic_rk4.tex}}};


% Labels positioned relative to nearest plots
% Column headers
\\node (col1header) [above=0cm of euler-torch] {\\rowlabel{torchdiffeq}};
\\node (col2header) [above=0cm of euler-mp] {\\rowlabel{rampde}};

% Row labels
\\node (euler-label) [left=0cm of euler-torch] {\\rotatebox{90}{\\rowlabel{Euler}}};
\\node (rk4-label) [left=0cm of rk4-torch] {\\rotatebox{90}{\\rowlabel{RK4}}};

% Add overall axis labels (since only the bottom-right plot has them)

% \\node[below=1.2cm of rk4-torch, font=\\Large] {\\textbf{Number of Time Steps}};
% \\node[left=2.5cm of eler-torch, rotate=90, font=\\Large] {\\textbf{Relative Error}};

\\end{tikzpicture}"""

    # Save combined TikZ file
    combined_path = output_dir / 'cnf_roundoff_combined_2x2.tex'
    with open(combined_path, 'w') as f:
        f.write(combined_content)

    # Create standalone LaTeX file - exact copy of original format
    standalone_content = f"""\\documentclass{{standalone}}
\\usepackage{{pgfplots}}
\\usepackage{{xcolor}}
\\usepackage{{tikz}}
\\usetikzlibrary{{positioning,calc}}
\\pgfplotsset{{compat=1.17}}

% convenient boxed label macro (following project conventions)
\\newcommand{{\\labelbox}}[1]{{%
  \\tikz[baseline]{{\\node[draw=gray, rounded corners=3pt, fill=gray!15, inner sep=4pt, text depth=0pt]{{\\bfseries #1}};}}%
}}

% Row label macro for solver names
\\newcommand{{\\rowlabel}}[1]{{%
  \\tikz[baseline]{{\\node[draw=black, rounded corners=3pt, fill=black, inner sep=6pt, text depth=0pt, text=white]{{\\bfseries #1}};}}%
}}

\\begin{{document}}
{combined_content}
\\end{{document}}"""

    standalone_path = output_dir / 'cnf_roundoff_combined_2x2_standalone.tex'
    with open(standalone_path, 'w', newline='') as f:
        # Convert to Windows line endings and add two extra blank lines at end
        windows_content = standalone_content.replace('\n', '\r\n') + '\r\n\r\n\r\n'
        f.write(windows_content)

    print(f"Created combined 2x2 TikZ file: {combined_path}")
    print(f"Created combined 2x2 standalone file: {standalone_path}")

    return combined_path


def main():
    parser = argparse.ArgumentParser(
        description='Create TikZ roundoff error plots for CNF experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot for rampde+fp16+dynamic+rk4
  python plot_cnf_roundoff.py --method rk4 --precision float16 --odeint rampde --scaler dynamic

  # Plot for torchdiffeq+fp16+grad+rk4
  python plot_cnf_roundoff.py --method rk4 --precision float16 --odeint torchdiffeq --scaler grad

  # Plot for configurations without scaler (none)
  python plot_cnf_roundoff.py --method rk4 --precision float16 --odeint torchdiffeq --scaler none

  # Create all four individual plots needed for 2x2 combined figure
  python plot_cnf_roundoff.py --create-combined
        """
    )
    parser.add_argument('--csv-path', type=str,
                       default='raw_data/cnf_roundoff_results.csv',
                       help='Path to CNF roundoff results CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/fig_cnf_roundoff',
                       help='Output directory for generated files')
    parser.add_argument('--method', type=str,
                       choices=['euler', 'rk4'],
                       help='ODE solver method')
    parser.add_argument('--precision', type=str,
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Precision type')
    parser.add_argument('--odeint', type=str,
                       choices=['torchdiffeq', 'rampde'],
                       help='ODE integration library')
    parser.add_argument('--scaler', type=str,
                       choices=['none', 'grad', 'dynamic'],
                       help='Scaler type')
    parser.add_argument('--no-legend', action='store_true',
                       help='Hide legend in the plot')
    parser.add_argument('--no-xlabel', action='store_true',
                       help='Hide x-axis label')
    parser.add_argument('--no-ylabel', action='store_true',
                       help='Hide y-axis label')
    parser.add_argument('--create-combined', action='store_true',
                       help='Create all four individual plots for 2x2 combined figure and generate combined figure')

    args = parser.parse_args()

    # Validate required arguments for single plot mode
    if not args.create_combined:
        required_args = ['method', 'precision', 'odeint', 'scaler']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(f"The following arguments are required for single plot mode: {', '.join('--' + arg for arg in missing_args)}")

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle combined figure creation
    if args.create_combined:
        print(f"Creating combined 2x2 figure with all four configurations")
        print(f"Reading from: {csv_path}")
        print(f"Output directory: {output_dir}")
        print()

        # Define the four configurations needed for the 2x2 grid
        configs = [
            ("euler", "float16", "torchdiffeq", "grad"),
            ("euler", "float16", "rampde", "dynamic"),
            ("rk4", "float16", "torchdiffeq", "grad"),
            ("rk4", "float16", "rampde", "dynamic")
        ]

        # Create all four individual plots
        for i, (method, precision, odeint, scaler) in enumerate(configs):
            config_str = f"{odeint}_{precision}_{scaler}_{method}"
            print(f"Creating plot {i+1}/4: {config_str}")

            # Load and process data
            data, error_columns = load_cnf_roundoff_data(
                csv_path, method, precision, odeint, scaler
            )

            # Determine which labels to show based on position in 2x2 grid
            # Only the bottom-right plot (index 3) should have labels and legend
            show_legend = (i == 3)  # Only bottom-right has legend
            show_xlabel = (i == 3)  # Only bottom-right
            show_ylabel = (i == 3)  # Only bottom-right

            # Create TikZ plot
            tikz_path = create_tikz_plot(
                data, error_columns, output_dir, config_str,
                show_legend=show_legend,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel
            )

        # Create combined 2x2 figure
        print("\nCreating combined 2x2 figure...")
        combined_path = create_combined_2x2_figure(output_dir)

        print(f"\nAll plots created successfully!")
        print(f"Combined 2x2 figure: {combined_path}")
        print(f"\nTo compile the combined figure:")
        print(f"  cd {output_dir}")
        print(f"  pdflatex cnf_roundoff_combined_2x2_standalone.tex")

        return

    # Handle single plot creation (original behavior)
    # Create configuration string for filenames
    config_str = f"{args.odeint}_{args.precision}_{args.scaler}_{args.method}"

    print(f"Reading from: {csv_path}")
    print(f"Configuration: {args.method}, {args.precision}, {args.odeint}, {args.scaler}")
    print(f"Output directory: {output_dir}")
    print()

    # Load and process data
    data, error_columns = load_cnf_roundoff_data(
        csv_path, args.method, args.precision, args.odeint, args.scaler
    )

    # Create TikZ plot
    tikz_path = create_tikz_plot(
        data, error_columns, output_dir, config_str,
        show_legend=not args.no_legend,
        show_xlabel=not args.no_xlabel,
        show_ylabel=not args.no_ylabel
    )

    print(f"\nTikZ plot ready: {tikz_path}")
    print(f"\nTo compile the standalone version:")
    print(f"  cd {output_dir}")
    print(f"  pdflatex cnf_roundoff_{config_str}_standalone.tex")


if __name__ == "__main__":
    main()
