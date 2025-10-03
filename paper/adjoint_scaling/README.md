# Adjoint Scaling Experiment

Tests gradient accuracy of mixed-precision ODE solvers with and without dynamic loss scaling.

## Usage

```bash
# Run locally (requires GPU)
python run_adjoint_scaling.py

# Run via SLURM
bash run_slurm.sh
```

## Outputs

- `polynomial_state.png` - ODE solution trajectory (log scale)
- `polynomial_velocity.png` - Velocity magnitude (log scale)
- `state_curve.csv` - State values over time
- `velocity_curve.csv` - Velocity values over time
- `run_info.txt` - Gradient accuracy table and metadata
