# OTFlowLarge Experiment

Large-scale optimal transport flow with mixed precision evaluation.

## Overview

This experiment tests:
- Optimal transport between high-dimensional distributions
- `torchdiffeq` vs `rampde` ODE solvers
- Different precision modes (float32, float16, bfloat16)
- Gradient scaling strategies
- MMD (Maximum Mean Discrepancy) loss

Problem: Transport samples from mixture of Gaussians to moon distribution in high dimensions

## Dataset Setup

The experiment uses the BSDS300 dataset from density estimation literature:

- **miniboone**: Included in `data/miniboone/` (~12MB)
- **BSDS300**: Must be downloaded (~20MB)

### Download BSDS300

Run the download script:

```bash
python download_datasets.py
```

This will download and extract the BSDS300 dataset from [Zenodo](https://zenodo.org/record/1161203) (preprocessed data from Papamakarios et al.'s MAF paper).

**Manual alternative**: Download `BSDS300.zip` from Zenodo and extract to `data/BSDS300/BSDS300.hdf5`

## Files

- `otflowlarge.py`: Main optimal transport training script
- `Phi.py`: Potential function network architecture
- `mmd.py`: Maximum Mean Discrepancy loss implementation
- `datasets/`: Dataset loading modules for BSDS300, POWER, GAS, HEPMASS, MINIBOONE
- Supporting utility files (`.py` files in directory)
- `run_experiment.sh`: Full experiment runner (submits SLURM jobs)
- `run_test.sh`: Quick test runner (local execution, reduced iterations)
- `job_otflowlarge.sbatch`: SLURM batch job template
- `generate_otflowlarge_table.py`: Generate performance comparison table

## Quick Test

```bash
./run_test.sh
```

**Expected runtime**: ~15-25 minutes per job
**What it does**:
- Runs 500 iterations (vs 5000+ in production)
- Tests single dataset (bsds300)
- All precision and scaling combinations
- Validates every 50 iterations (10 validation points)
- Generates data for performance table

## Full Experiment

```bash
./run_experiment.sh
```

**Expected runtime**: ~6-10 hours total (parallel SLURM jobs)
**What it does**: Trains optimal transport models for 5000 iterations across precision/solver combinations

## Expected Outputs

### Raw Data (`raw_data/`)
- Experiment directories named by configuration (e.g., `otflowlarge_rampde_float32_noseed/`)
- Each contains:
  - `summary_otflowlarge.csv`: Aggregated metrics
  - `mmd_loss.txt`: MMD loss over training
  - `final_samples.npy`: Final transported samples
  - `config.json`: Experiment configuration

### Processed Outputs (`outputs/`)

After running processing script:

```bash
python generate_otflowlarge_table.py
```

**Tables** (`outputs/tab_otflowlarge_results/`):
- `otflowlarge_results_table.tex`: LaTeX performance table
- `otflowlarge_table_standalone.tex`: Standalone compilable version

## Configuration

Edit `run_experiment.sh` to modify:
- Precision modes: uncomment/comment configurations
- Iterations: `--niters` argument
- Batch size: `--batch-size` argument
- Learning rate: `--lr` argument
- Hidden dimensions: `--hidden-dim` argument

Default hyperparameters:
- Iterations: 5000
- Batch size: 512
- Learning rate: 0.001
- Hidden dim: 64
- Time steps: 20
- Data dimensions: varies by problem

## Notes

- Test script uses reduced iterations (100) for quick validation
- No fixed random seed by default (stochastic comparison)
- SLURM account `mathg3` is hardcoded
- Results saved to `./raw_data` by default
- Requires GPU and conda environment `torch26`
- Large memory requirement due to high-dimensional data