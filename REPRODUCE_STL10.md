# Instructions for Reproducing STL10 Experiments

This document provides step-by-step instructions for setting up and running the STL10 experiments on a different machine to verify memory usage measurements.

## Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX A6000)
- Miniconda installed in `${HOME}/miniconda3`
- Git access to the rampde repository

## Setup Instructions

### 1. Clone and Checkout the Paper Version

```bash
cd ${HOME}
git clone https://github.com/EmoryMLIP/rampde.git
cd rampde
git checkout v0.1.0-paper
```

### 2. Create Conda Environment

```bash
conda create -n torch28 python=3.11 -y
conda activate torch28
```

### 3. Install Dependencies

```bash
pip install -r requirements-paper.txt
pip install -e .
```

### 4. Verify Package Versions

**IMPORTANT**: Before running experiments, verify that the installed versions exactly match:

```bash
python -c "import torch; import numpy; import torchdiffeq; print(f'torch: {torch.__version__}'); print(f'numpy: {numpy.__version__}'); print(f'torchdiffeq: {torchdiffeq.__version__}')"
```

**Expected output:**
```
torch: 2.8.0+cu126
numpy: 2.3.3
torchdiffeq: 0.2.5
```

If versions don't match exactly, stop and investigate why.

### 5. Verify CUDA and GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Running STL10 Experiments

### Test Run (Quick Verification)

First, do a quick test to ensure everything works:

```bash
cd paper/stl10
python ode_stl10.py \
  --precision float32 \
  --odeint rampde \
  --nepochs 1 \
  --batch_size 16 \
  --width 128 \
  --seed 25 \
  --stable \
  --results_dir ./test_results
```

### Full Experiment: Float32 with rampde

This is the key experiment to compare memory usage:

```bash
cd paper/stl10
mkdir -p results
python ode_stl10.py \
  --precision float32 \
  --odeint rampde \
  --nepochs 160 \
  --batch_size 16 \
  --width 128 \
  --seed 25 \
  --stable \
  --results_dir ./results
```

**Expected runtime:** ~3-4 hours on RTX A6000

### Full Experiment: Float16 with rampde (no scaler)

```bash
cd paper/stl10
python ode_stl10.py \
  --precision float16 \
  --odeint rampde \
  --nepochs 160 \
  --batch_size 16 \
  --width 128 \
  --seed 25 \
  --stable \
  --no_grad_scaler \
  --no_dynamic_scaler \
  --results_dir ./results
```

## Analyzing Results

### 1. Check Memory Usage

After experiments complete, examine the CSV files in the results directories:

```bash
# For float32 run
tail -5 results/stl10_float32_rampde_*/stl10_*.csv | grep -v "^$"

# For float16 run
tail -5 results/stl10_float16_none_rampde_*/stl10_*.csv | grep -v "^$"
```

Look at the `max_memory_mb` column (last column).

### 2. Compare with Original Results

**Original measurements (on cluster):**
- float32 + rampde: 12,131 MB (11.85 GB)
- float16 + rampde: 2,304 MB (2.25 GB)
- **Ratio: 5.26×**

**Your measurements (RTX A6000):**
- float32 + rampde: ??? MB
- float16 + rampde: ??? MB
- **Ratio: ???**

Report back whether you observe:
- Similar ~5× reduction (consistent with cluster)
- Smaller ~2× reduction (more aligned with theoretical expectation)

### 3. Create Summary

```bash
cd paper/stl10
python aggregate_stl10_results.py --results_dir ./results
```

This will create a summary CSV with all metrics.

## Troubleshooting

### Package Version Mismatch

If package versions don't match, try:

```bash
pip install torch==2.8.0 numpy==2.3.3 torchdiffeq==0.2.5 --force-reinstall
```

### Out of Memory Errors

If you get OOM errors with batch_size=16, try reducing to batch_size=8:

```bash
python ode_stl10.py --batch_size 8 [other args...]
```

### CUDA Errors

Ensure CUDA drivers are up to date and compatible with PyTorch 2.8.0 (requires CUDA 12.6).

## Questions to Answer

After running the experiments, please report:

1. What are the exact `max_memory_mb` values for:
   - float32 + rampde
   - float16 + rampde (none scaler)

2. What is the memory reduction ratio (fp32 / fp16)?

3. What GPU model was used?

4. Did the package versions match exactly?

5. What were the final validation accuracies?

This information will help determine whether the 5× memory reduction observed on the cluster is GPU-specific or due to other factors.
