# Paper Experiments

This directory contains all code, scripts, and instructions to reproduce the experiments in the rampde paper.

## Structure

Each experiment has its own subdirectory with:

- **Experiment code**: Python files to run the experiment
- **Run scripts**:
  - `run_experiment.sh`: Full experiment (for paper results)
  - `run_test.sh`: Quick validation test (2-3 epochs/iterations)
- **SLURM job files**: `.sbatch` files for HPC execution
- **Processing scripts**: Python scripts to generate figures and tables
- **Data directories**:
  - `raw_data/`: Experiment outputs (created by run scripts)
  - `outputs/`: Paper-ready LaTeX figures and tables (created by processing scripts)

## Experiments

### 1. CNF (Continuous Normalizing Flows)

**Directory**: `cnf/`
**Purpose**: Compare gradient scaling approaches across different precision modes for continuous normalizing flow models on toy datasets.

**Outputs**:

- `fig_cnf_overview/`: LaTeX figure showing CNF results across datasets
- `tab_cnf_results/`: LaTeX table of numerical results

### 2. Roundoff Error Analysis

**Directory**: `roundoff/`
**Purpose**: Analyze roundoff error propagation in CNF with different precision modes.

**Outputs**:

- `fig_cnf_roundoff/`: Roundoff error visualization plots

### 3. OTFlowLarge (Large-Scale Optimal Transport Flow)

**Directory**: `otflowlarge/`
**Purpose**: Test large-scale optimal transport flow problems with different precision and scaling settings.

**Outputs**:

- `tab_otflowlarge_results/`: Performance metrics table

### 4. STL10 Image Classification

**Directory**: `stl10/`
**Purpose**: Evaluate neural ODE performance on STL10 image classification across precision configurations.

**Outputs**:

- `fig_stl10_train_loss/`: Training convergence plots
- `tab_stl10_results/`: Performance comparison table

## Running Experiments

### Quick Test (Recommended First)

Test scripts run reduced-iteration versions that generate enough data for figure/table creation:

```bash
cd paper/cnf
./run_test.sh          # ~5-8 min/job, 50 iters, generates visualizations

cd ../stl10
./run_test.sh          # ~15-20 min/job, 3 epochs, convergence data

cd ../otflowlarge
./run_test.sh          # ~5-10 min/job, 100 iters, table data

cd ../roundoff
./run_test.sh          # ~8-10 min, 300 iters, error metrics
```

**Key features of test runs**:

- Reduced iterations but sufficient data points for plotting
- Use different random seeds than production (distinguishable)
- Output to `./raw_data` in each experiment directory
- Can generate figures/tables from test data to verify processing scripts

### Full Experiments

For paper results:

```bash
cd paper/cnf
./run_experiment.sh

cd ../stl10
./run_experiment.sh

cd ../otflowlarge
./run_experiment.sh

cd ../roundoff
./run_experiment.sh
```

### Generating Figures and Tables

After experiments complete, each experiment has a `process_results.sh` script that generates all figures and tables for that experiment:

```bash
# Process individual experiments
cd paper/cnf
./process_results.sh          # Generates Figure 2 + Table 2

cd ../stl10
./process_results.sh          # Generates Figure 4 + summary table

cd ../otflowlarge
./process_results.sh          # Generates Table 3

cd ../roundoff
./process_results.sh          # Generates Figure 3 (roundoff plots)
```

**Or process all experiments at once:**

```bash
cd paper
python process_all_results.py                    # Process all experiments
python process_all_results.py --skip-tables      # Skip table generation
python process_all_results.py --experiments cnf,stl10  # Process only CNF and STL10
```

**Outputs** are saved to `[experiment]/outputs/` in each experiment directory.

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **SLURM**: HPC cluster with SLURM job scheduler
- **Conda**: Conda environment `torch28`

### Hardware Specifications

All experiments and runtime estimates in this documentation were performed on:

- **GPU**: NVIDIA RTX A6000 (48GB VRAM, Ampere architecture)
- **CUDA**: Compatible CUDA version with PyTorch 2.8

Performance results may vary depending on your hardware configuration.

## Environment Setup

```bash
# Activate conda environment
conda activate torch28

# Verify rampde is installed
python -c "from rampde import odeint; print('rampde installed successfully')"
```

## Dependencies

Core dependencies are in `requirements-paper.txt`. Install with:

```bash
pip install -r requirements-paper.txt
```

## Code Organization

### Utility Modules

- **`experiment_runtime.py`**: Runtime utilities for RUNNING experiments

  - Environment setup and ODE solver imports
  - Precision configuration (float32, tfloat32, float16, bfloat16)
  - Gradient scaler setup (GradScaler, DynamicScaler)
  - Experiment directory creation and logging
  - Training utility classes (RunningAverageMeter, etc.)
  - **Used by**: cnf.py, ode_stl10.py, otflowlarge.py, roundoff_cnf.py

- **`analysis_utils.py`**: Analysis utilities for PROCESSING results
  - Parsing experiment directory names
  - Loading experiment results from CSV files
  - Creating legend labels for plots
  - **Used by**: Processing scripts that generate figures and tables

### File Structure

```
paper/
├── experiment_runtime.py      # Runtime utilities for experiments
├── analysis_utils.py           # Analysis utilities for processing results
├── process_all_results.py      # Master processing script
├── .gitignore                  # Excludes raw_data/ and outputs_backup/
├── cnf/
│   ├── cnf.py                  # Main experiment script
│   ├── run_experiment.sh       # Full experiment runner
│   ├── run_test.sh             # Quick test runner
│   ├── process_results.sh      # Results processing script
│   ├── raw_data/               # Generated experiment data (gitignored)
│   └── outputs/                # Generated figures and tables
├── stl10/                      # (similar structure as cnf/)
├── otflowlarge/                # (similar structure as cnf/)
└── roundoff/                   # (similar structure as cnf/)
```

## Expected Runtimes

**Note**: All runtime estimates are based on NVIDIA RTX A6000 (48GB) GPU. Actual runtimes may vary depending on your hardware.

### Test Runs (Reduced Iterations)

- **CNF test**: ~5-8 minutes per job (50 iters, 10 validation points)
- **STL10 test**: ~15-20 minutes per job (3 epochs)
- **OTFlowLarge test**: ~5-10 minutes per job (100 iters, 4 validation points)
- **Roundoff test**: ~8-10 minutes (300 iters)

### Full Production Runs

- **CNF full**: ~2-4 hours total (2000 iters × multiple configs)
- **STL10 full**: ~8-12 hours per configuration (160 epochs)
- **OTFlowLarge full**: ~6-10 hours total (5000+ iters × multiple configs)
- **Roundoff full**: ~30-60 minutes (2000 iters)

## Notes

- Test scripts use different random seeds than production runs to distinguish test results
- Raw data directories may be large (several GB)
- SLURM job logs are saved to `slurm_logs/` in each experiment directory
- All paths in processing scripts are relative to the experiment directory

## See Also

- Individual experiment README files for detailed instructions
- Root repository CLAUDE.md for development environment setup
