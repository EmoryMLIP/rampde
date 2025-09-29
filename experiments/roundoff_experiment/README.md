# Roundoff Error Experiments

This directory contains lightweight experiments to measure roundoff errors in neural ODE integration as step size h approaches zero. The experiments validate the theoretical analysis showing that rampde's mixed-precision approach maintains better numerical accuracy than standard implementations.

## Overview

The experiments compare:
- **Precisions**: float16, bfloat16 (against fp64 reference)
- **Backends**: torchdiffeq, rampde
- **Scalers**: GradScaler, DynamicScaler, no scaling
- **Step sizes**: h ∈ {0.1, 0.01, 0.001, 0.0001, 0.00001}
- **Methods**: Euler, RK4

Each experiment uses the exact hyperparameters from the main experiments to ensure reproducibility.

## Experiments

### 1. CNF (Continuous Normalizing Flows)
- **Model**: hidden_dim=32, width=128
- **Dataset**: 8gaussians (single batch of 128 samples)
- **Measures**: State error, log-determinant Jacobian error, gradient errors

### 2. OTFlow (Optimal Transport Flow)
- **Model**: hidden_dim=256, alpha=[1.0, 100.0, 15.0]
- **Dataset**: BSDS300 (single batch of 512 samples)
- **Measures**: Transport map error, potential gradient errors


## Running the Experiments

### Quick Start

```bash
# Run all experiments
./run_roundoff.sh

# Or run individual experiments
python roundoff_cnf.py
python roundoff_otflow.py

# Generate plots
python plot_roundoff.py
```

### Expected Runtime
- CNF: ~3-5 minutes
- OTFlow: ~10-15 minutes  
- Total: ~15-20 minutes

## Implementation Details

### Determinism Check
The framework automatically detects non-deterministic computations by running each configuration twice. If results differ, it runs 10 iterations and reports mean ± std. This is particularly important for operations that may have non-deterministic behavior in low precision.

### Error Metrics
For each configuration, we measure:
1. **Solution error**: ||y_fp16(T) - y_fp64(T)|| / ||y_fp64(T)||
2. **Gradient error**: ||∇L_fp16 - ∇L_fp64|| / ||∇L_fp64||
3. **Parameter gradient errors**: Similar relative errors for model parameters

### Key Files
- `roundoff_analyzer.py`: Base class for roundoff analysis
- `roundoff_*.py`: Experiment-specific implementations
- `run_roundoff.sh`: Unified runner script
- `plot_roundoff.py`: Visualization and analysis

## Expected Results

The experiments should demonstrate:

1. **Roundoff error growth**: As h→0, roundoff errors dominate truncation errors
2. **rampde advantage**: Better error control with DynamicScaler
3. **Precision comparison**: Different behavior between float16 and bfloat16
4. **Method differences**: RK4 vs Euler error patterns

## Output Structure

```
results/
├── cnf_roundoff_results.csv      # Raw results for CNF
├── otflow_roundoff_results.csv    # Raw results for OTFlow
├── plots/
│   ├── error_vs_h_*.pdf         # Individual error plots
│   └── scaler_comparison_all.pdf # Comparison across experiments
└── summary_statistics.txt         # Summary of key findings
```

## Interpreting Results

The log-log plots should show:
- Error decreasing with h until roundoff effects dominate
- rampde+DynamicScaler maintaining lower errors at small h
- Reference lines showing theoretical convergence rates (h^p)

This provides empirical validation of the theoretical roundoff analysis in the paper.