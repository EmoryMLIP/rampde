#!/usr/bin/env python3
"""
Generate CNF overview figure LaTeX code (wide/transpose version).

This script generates the complete LaTeX code for the wide CNF overview figure
showing results in a transposed layout with precision configurations as rows
and dataset/solver combinations as columns.
"""

import os

def generate_latex_figure_wide():
    """Generate the complete LaTeX figure code for the wide version."""
    latex_content = r"""\documentclass{standalone}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,calc}

% convenient boxed label macro (following project conventions)
\newcommand{\labelbox}[1]{%
  \tikz[baseline]{\node[draw=gray, rounded corners=3pt, fill=gray!15, inner sep=4pt, text depth=0pt]{\Large\bfseries #1};}%
}


% Row label macro for solver names
\newcommand{\rowlabel}[1]{%
  \tikz[baseline]{\node[draw=black, rounded corners=3pt, fill=black, inner sep=6pt, text depth=0pt, text=white]{\large\bfseries #1};}%
}

\begin{document}
\begin{tikzpicture}[node distance=0.3cm]

% Define parameters
\def\imgwidth{3.8cm}
\def\imgheight{3.8cm}
\def\colsep{1.2cm}

% Column headers at top - solver combinations only
\node (col1header) {\rowlabel{torchdiffeq}};
\node (col2header) [right=\colsep of col1header] {\rowlabel{rampde}};
\node (col3header) [right=\colsep of col2header] {\rowlabel{torchdiffeq}};
\node (col4header) [right=\colsep of col3header] {\rowlabel{rampde}};
\node (col5header) [right=\colsep of col4header] {\rowlabel{torchdiffeq}};
\node (col6header) [right=\colsep of col5header] {\rowlabel{rampde}};

% First row: bfloat16
\node (bf16-col1) [below=-0.2cm of col1header] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_bfloat16_torchdiffeq-samples.jpg}};
\node (bf16-label) [left=-0.2cm of bf16-col1] {\rotatebox{90}{\rowlabel{bfloat16}}};
\node (bf16-col2) [below=-0.2cm of col2header] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_bfloat16_rampde-samples.jpg}};
\node (bf16-col3) [below=-0.2cm of col3header] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_bfloat16_torchdiffeq-samples.jpg}};
\node (bf16-col4) [below=-0.2cm of col4header] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_bfloat16_rampde-samples.jpg}};
\node (bf16-col5) [below=-0.2cm of col5header] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_bfloat16_torchdiffeq-samples.jpg}};
\node (bf16-col6) [below=-0.2cm of col6header] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_bfloat16_rampde-samples.jpg}};

% Second row: float16 none
\node (f16none-col1) [below=-0.2cm of bf16-col1] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_float16_none_torchdiffeq-samples.jpg}};
\node (f16none-label) [left=-0.2cm of f16none-col1] {\rotatebox{90}{\rowlabel{float16 none}}};
\node (f16none-col2) [below=-0.2cm of bf16-col2] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_float16_none_rampde-samples.jpg}};
\node (f16none-col3) [below=-0.2cm of bf16-col3] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_float16_none_torchdiffeq-samples.jpg}};
\node (f16none-col4) [below=-0.2cm of bf16-col4] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_float16_none_rampde-samples.jpg}};
\node (f16none-col5) [below=-0.2cm of bf16-col5] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_float16_none_torchdiffeq-samples.jpg}};
\node (f16none-col6) [below=-0.2cm of bf16-col6] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_float16_none_rampde-samples.jpg}};

% Third row: float16 grad
\node (f16grad-col1) [below=-0.2cm of f16none-col1] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_float16_grad_torchdiffeq-samples.jpg}};
\node (f16grad-label) [left=-0.2cm of f16grad-col1] {\rotatebox{90}{\rowlabel{float16 grad}}};
\node (f16grad-col2) [below=-0.2cm of f16none-col2] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_float16_grad_rampde-samples.jpg}};
\node (f16grad-col3) [below=-0.2cm of f16none-col3] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_float16_grad_torchdiffeq-samples.jpg}};
\node (f16grad-col4) [below=-0.2cm of f16none-col4] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_float16_grad_rampde-samples.jpg}};
\node (f16grad-col5) [below=-0.2cm of f16none-col5] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_float16_grad_torchdiffeq-samples.jpg}};
\node (f16grad-col6) [below=-0.2cm of f16none-col6] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_float16_grad_rampde-samples.jpg}};

% Fourth row: float16 dynamic / target
\node (f16dyn-col1) [below=-0.2cm of f16grad-col1] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_float32-target.jpg}};
% \node (f16dyn-label) [left=0.2cm of f16dyn-col1] {\rotatebox{90}{\rowlabel{target}}};
\node (f16dyn-col1-label) [above=-8mm of f16dyn-col1.north] {\rowlabel{target}};
\node (f16dyn-col2) [below=-0.2cm of f16grad-col2] {\includegraphics[width=\imgwidth,height=\imgheight]{2spirals_float16_dynamic_rampde-samples.jpg}};
\node (f16dyn-col2-label) [above=-8mm of f16dyn-col2.north] {\rowlabel{float16 dynamic}};
\node (f16dyn-col3) [below=-0.2cm of f16grad-col3] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_float32-target.jpg}};
\node (f16dyn-col3-label) [above=-8mm of f16dyn-col3.north] {\rowlabel{target}};
\node (f16dyn-col4) [below=-0.2cm of f16grad-col4] {\includegraphics[width=\imgwidth,height=\imgheight]{8gaussians_float16_dynamic_rampde-samples.jpg}};
\node (f16dyn-col4-label) [above=-8mm of f16dyn-col4.north] {\rowlabel{float16 dynamic}};
\node (f16dyn-col5) [below=-0.2cm of f16grad-col5] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_float32-target.jpg}};
\node (f16dyn-col5-label) [above=-8mm of f16dyn-col5.north] {\rowlabel{target}};
\node (f16dyn-col6) [below=-0.2cm of f16grad-col6] {\includegraphics[width=\imgwidth,height=\imgheight]{checkerboard_float16_dynamic_rampde-samples.jpg}};
\node (f16dyn-col6-label) [above=-8mm of f16dyn-col6.north] {\rowlabel{float16 dynamic}};


\end{tikzpicture}
\end{document}"""
    
    return latex_content

def main():
    """Main function to generate and save the wide LaTeX figure."""
    # Define output path - use relative path to outputs directory with LaTeX label
    output_path = "outputs/fig_cnf_overview"
    output_file = os.path.join(output_path, "cnf_overview_figure_wide.tex")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate LaTeX content
    latex_content = generate_latex_figure_wide()
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"Generated wide LaTeX figure: {output_file}")
    print("The figure references images in the same directory as the LaTeX file.")
    print("Layout: 4 precision rows × 6 dataset/solver columns")

if __name__ == "__main__":
    main()