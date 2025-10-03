# CNF Experiment

Continuous Normalizing Flow gradient scaling comparison across toy datasets.

## Overview

This experiment compares:
- `torchdiffeq` vs `rampde` ODE solvers
- Different precision modes (float32, tfloat32, bfloat16, float16)
- Gradient scaling strategies (none, grad scaler only, dynamic scaler, both)

Datasets: `checkerboard`, `8gaussians`, `2spirals`

## Files

- `cnf.py`: Main CNF training script
- `toy_data.py`: Toy dataset generation
- `run_experiment.sh`: Full experiment runner (submits SLURM jobs)
- `run_test.sh`: Quick test runner (local execution, reduced iterations)
- `job_cnf.sbatch`: SLURM batch job template
- `generate_cnf_overview_wide.py`: Generate LaTeX figure
- `extract_cnf_subplots.py`: Extract individual subplots
- `generate_experiment_table.py`: Generate numerical results table

## Quick Test

```bash
./run_test.sh
```

**Expected runtime**: ~5-8 minutes per job
**What it does**:
- Runs 200 iterations (vs 2000 in production)
- Tests single dataset (8gaussians)
- All precision and scaling combinations
- Generates 10 validation points for plotting
- Creates visualizations for figure generation

## Full Experiment

```bash
./run_experiment.sh
```

**Expected runtime**: ~2-4 hours total (parallel SLURM jobs)
**What it does**: Submits SLURM jobs for all dataset/precision/solver combinations

## Expected Outputs

### Raw Data (`raw_data/`)
- Experiment directories named by configuration (e.g., `checkerboard_rk4_rampde_tfloat32_noscaling_seed42/`)
- Each contains:
  - `summary.csv`: Loss metrics over training
  - `densities_*.png`: Learned density visualizations
  - `config.json`: Experiment configuration

### Processed Outputs (`outputs/`)

After running processing scripts:

```bash
# Process results from a specific seed (default: 24 for test runs)
./process_results.sh 24  # Process test runs
./process_results.sh 42  # Process production runs

# Or run individual scripts:
python aggregate_cnf_results.py --raw-data-dir raw_data/cnf --seed 24
python generate_cnf_overview_wide.py
python generate_experiment_table.py --experiment-type cnf --input raw_data/cnf/summary_cnf.csv
```

**Figures** (`outputs/fig_cnf_overview/`):
- `cnf_overview_figure_wide.tex`: Main LaTeX figure
- `cnf_overview_figure_wide.pdf`: Compiled PDF (automatically generated)

**Tables** (`outputs/tab_cnf_results/`):
- `cnf_results_table.tex`: LaTeX table of numerical results
- `cnf_table_standalone.tex`: Standalone compilable version
- `cnf_table_standalone.pdf`: Compiled PDF (automatically generated)

**CSV Data** (`outputs/`):
- `cnf_results_seed{N}.csv`: Summary of all experimental results for the specified seed

## Configuration

Edit `run_experiment.sh` to modify:
- Datasets: `datasets` array
- Precision modes: `precision` in loops
- Training hyperparameters: `dataset_args` dictionary
- Seed: `seed` variable

## Seed Management

The scripts support filtering results by random seed:
- **Test runs** use `seed=24` (quick validation with 50 iterations)
- **Production runs** use `seed=42` (full 2000 iterations)

To process specific seeds:
```bash
./process_results.sh 24  # Default: process test runs
./process_results.sh 42  # Process production runs when complete
```

## Notes

- SLURM account `mathg3` is hardcoded
- Results saved to `./raw_data` by default
- Requires GPU and conda environment `torch26` (torchdiffeq) or `torch27` (rampde)