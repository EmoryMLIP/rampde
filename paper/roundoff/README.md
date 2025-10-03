# Roundoff Error Analysis Experiment

Analyze roundoff error propagation in CNF models across precision modes.

## Overview

This experiment analyzes:
- Roundoff error accumulation during ODE integration
- Error breakdown by precision (float64, float32, float16, bfloat16)
- Comparison between Euler and RK4 solvers
- Absolute vs relative error metrics

Dataset: 8gaussians toy distribution

## Files

- `roundoff_cnf.py`: Main roundoff analysis script
- `roundoff_analyzer.py`: Error analysis utilities
- Additional plotting utilities (`.py` files in directory)
- `run_experiment.sh`: Full experiment runner
- `run_test.sh`: Quick test runner (reduced iterations)
- `job_roundoff_cnf.sbatch`: SLURM batch job template
- `plot_cnf_roundoff.py`: Generate roundoff error plots

## Quick Test

```bash
./run_test.sh
```

**Expected runtime**: ~8-10 minutes
**What it does**:
- Runs 300 iterations (vs 2000 in production)
- Compares all precision modes against float64 reference
- Generates error metrics CSV
- Enough data points to visualize error trends

## Full Experiment

```bash
./run_experiment.sh
```

**Expected runtime**: ~30-60 minutes
**What it does**: Runs comprehensive roundoff analysis with multiple precision modes

## Expected Outputs

### Raw Data (`raw_data/`)
- `cnf_roundoff_results.csv`: Aggregated roundoff error metrics
- Contains columns:
  - `method`: Solver (euler, rk4)
  - `precision`: Precision mode
  - `iteration`: Training iteration
  - `abs_error_mean`, `abs_error_std`: Absolute error statistics
  - `rel_error_mean`, `rel_error_std`: Relative error statistics

### Processed Outputs (`outputs/`)

After running processing script:

```bash
python plot_cnf_roundoff.py --create-combined
```

**Figures** (`outputs/fig_cnf_roundoff/`):
- Individual error plots per solver/precision
- `cnf_roundoff_combined_2x2.tex`: Combined 2×2 TikZ figure
- `cnf_roundoff_combined_2x2_standalone.tex`: Standalone compilable version
- `cnf_roundoff_combined_2x2.pdf`: Compiled PDF
- Supporting CSV data files

## Configuration

Edit `run_experiment.sh` to modify:
- Precision modes: uncomment/comment precision tests
- Solvers: modify method (`euler` or `rk4`)
- Iterations: `--niters` argument
- Dataset: `--data` argument

Default settings:
- Dataset: 8gaussians
- Iterations: 2000 (full), 100 (test)
- Reference precision: float64
- Comparison precisions: float32, float16, bfloat16

## Processing Script Options

```bash
# Generate specific plot
python plot_cnf_roundoff.py --method rk4 --precision float16

# Generate all individual plots
python plot_cnf_roundoff.py --create-all

# Generate 2×2 combined figure
python plot_cnf_roundoff.py --create-combined
```

## Notes

- Test script uses `seed=999` to distinguish from production `seed=42`
- Reference computation uses float64 for ground truth
- Errors computed as difference from float64 reference
- SLURM account `mathg3` is hardcoded
- Results saved to `./raw_data` by default
- Requires GPU and conda environment `torch26`