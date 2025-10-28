# STL10 Image Classification Experiment

Neural ODE image classification on STL10 dataset across precision configurations.

## Overview

This experiment evaluates:
- `torchdiffeq` vs `rampde` ODE solvers
- Different precision modes (float32, float16, bfloat16)
- Gradient scaling strategies for mixed precision

Dataset: STL10 (96×96 color images upsampled to 128x128 for better CUDA utilization, 10 classes)

## Files

- `ode_stl10.py`: Neural ODE classifier training script
- `run_experiment.sh`: Full experiment runner (submits SLURM jobs)
- `run_test.sh`: Quick test runner (local execution, 3 epochs)
- `job_ode_stl10.sbatch`: SLURM batch job template
- `plot_stl10_convergence.py`: Generate training convergence plots
- `generate_stl10_table.py`: Generate performance comparison table

## Quick Test

```bash
./run_test.sh
```

**Expected runtime**: ~15-20 minutes per job
**What it does**:
- Trains for 3 epochs (vs 100+ in production)
- Tests every epoch (3 validation points)
- All precision and scaling combinations
- Smaller batch size (16) for faster execution
- Generates data for convergence plots and tables

## Full Experiment

```bash
./run_experiment.sh
```

**Expected runtime**: ~3-12 hours per configuration depending on your GPU
**What it does**: Trains neural ODE classifier for 160 epochs across precision/solver combinations

## Evaluation (Computing Test Losses)

After training completes, evaluate the trained model checkpoints on the test set:

```bash
./run_evaluation.sh
```

**What it does**:
- Loads trained model checkpoints from `raw_data/ode_stl10/`
- Evaluates each checkpoint on the STL10 test set
- Computes and saves test losses and accuracies to `test_loss.txt` and `test_acc.txt`
- Required before running processing scripts to generate tables

**Note**: This step is necessary to populate the test metrics that appear in the final results table

## Expected Outputs

### Raw Data (`raw_data/`)
- Experiment directories named by configuration (e.g., `ode_stl10_rampde_float32_seed32_width128/`)
- Each contains:
  - `summary_ode_stl10.csv`: Aggregated metrics across runs
  - `train_loss.txt`, `val_loss.txt`, `val_acc.txt`: Training history
  - `test_loss.txt`, `test_acc.txt`: Final test metrics
  - `config.json`: Experiment configuration

### Processed Outputs (`outputs/`)

After running processing scripts:

```bash
python plot_stl10_convergence.py
python generate_stl10_table.py
```

**Figures** (`outputs/fig_stl10_train_loss/`):
- `stl10_train_loss_convergence.tex`: TikZ convergence plot
- `stl10_train_loss_convergence.pdf`: Compiled PDF
- Supporting CSV data files

**Tables** (`outputs/tab_stl10_results/`):
- `stl10_results_table.tex`: LaTeX performance table
- `stl10_results_table_standalone.tex`: Standalone compilable version

## Configuration

Edit `run_experiment.sh` to modify:
- Precision modes: uncomment/comment configurations
- Network width: `--width` argument
- Epochs: `--epochs` argument
- Learning rate: `--lr` argument
- Seed: `seed` variable

Default hyperparameters:
- Epochs: 100
- Batch size: 64
- Learning rate: 0.05
- Momentum: 0.9
- Weight decay: 1e-4
- Width: 128

## Notes

- Test script uses `seed=26` and 3 epochs for quick validation
- Production runs use `seed=25` and 160 epochs
- SLURM account configured in `job_ode_stl10.sbatch`
- Results saved to `./raw_data` by default
- Requires GPU, CUDA, and conda environment `torch28`
- STL10 dataset will be downloaded automatically to `~/data/stl10`