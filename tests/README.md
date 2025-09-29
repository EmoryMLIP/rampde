# Torchmpnode Test Suite

This folder contains unit tests for the rampde package. Below is a summary of each test and instructions for running the tests.

## Test Summaries

- **test_adjoint_scaling.py**: Checks the accuracy of gradients in low-precision time integration, especially with dynamic scaling for float16.
- **test_backward.py**: Verifies the accuracy of backward-mode derivatives (input, weights, time) using Taylor expansion tests.
- **test_odeint.py**: Confirms the convergence order of the custom ODE solvers (Euler and RK4) against analytical solutions.
- **test_rampde.py**: Compares solutions and gradients of rampde's ODE solver to torchdiffeq (if available) for both linear and neural ODEs.
- **test_rampde_tuple.py**: Tests tuple-valued ODEs for compatibility between rampde and torchdiffeq.
- **test_adjoint_scaling/**: (Directory) May contain additional or legacy tests for adjoint scaling.

## How to Run All Tests

From this directory, run:

```bash
python run_all_tests.py
```

Or, to run a specific test file:

```bash
python test_odeint.py
```

## Notes
- Some tests require a CUDA-capable GPU.
- `test_rampde.py` and `test_rampde_tuple.py` require `torchdiffeq` to be installed; they will be skipped if it is not available.
- Test output will summarize any failures or errors at the end.
