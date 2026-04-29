#!/bin/bash
# Compile the pgfplots figure in outputs/ from the CSVs in raw_data/.

set -euo pipefail
cd "$(dirname "$0")/outputs"

# Two passes: the shared legend uses \label/\ref across pgfplots.
pdflatex -interaction=nonstopmode fig_stability_probe.tex
pdflatex -interaction=nonstopmode fig_stability_probe.tex
