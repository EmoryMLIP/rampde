# rampde

rampde is a PyTorch-compatible library designed to provide automatic mixed-precision solvers for Neural Ordinary Differential Equations (ODEs). The package integrates seamlessly with PyTorch’s ecosystem, allowing users to replace standard solvers with mixed-precision alternatives for faster computation and reduced memory usage.

Key features include:

- Easy API compatibility with Pytorch's autocast and the torchdiffeq package.
- Support for both forward and backward computations with customizable precision.
- Benchmark tools for performance and memory profiling.
- Examples and tests for various neural ODE problems.

## Installation

Install the core package with:

```bash
pip install rampde
```

### Optional Dependencies

For benchmarking and comparison with torchdiffeq:

```bash
pip install "rampde[benchmarks]"
```

For development (includes testing and benchmark dependencies):

```bash
pip install "rampde[dev]"
```

For testing only:

```bash
pip install "rampde[testing]"
```

Note: `torchdiffeq` is an optional dependency. The core rampde functionality works without it. Install `torchdiffeq` separately if needed for comparisons.

## Quick Start

```python
import torch
from rampde import odeint

# Define your ODE function
class ODEFunc(torch.nn.Module):
    def forward(self, t, y):
        return -y

# Initial condition and time points
y0 = torch.tensor([1.0])
t = torch.linspace(0, 1, 10)

# Solve ODE numerically
solution = odeint(ODEFunc(), y0, t, method='rk4')
```

For mixed precision training, wrap the last line inside autocast

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    solution = odeint(func, y0, t, method='rk4')  # Automatically applies DynamicScaler
```

## Paper Reproducibility

This repository contains the experiments from the rampde paper:

```
@misc{celledoni2025mixedprecisiontrainingneural,
      title={Mixed Precision Training of Neural ODEs}, 
      author={Elena Celledoni and Brynjulf Owren and Lars Ruthotto and Tianjiao Nicole Yang},
      year={2025},
      eprint={2510.23498},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23498}, 
}
```

 All reproducibility scripts are in the `paper/` directory.

### Available Experiments

- **CNF (Continuous Normalizing Flows)**: Compare gradient scaling approaches across different precision modes ([paper/cnf/](paper/cnf/))
- **STL10 Image Classification**: Evaluate neural ODE performance on STL10 classification ([paper/stl10/](paper/stl10/))
- **OTFlowLarge (Optimal Transport Flow)**: Test large-scale optimal transport flow problems ([paper/otflowlarge/](paper/otflowlarge/))
- **Roundoff Error Analysis**: Analyze precision-related errors in ODE integration ([paper/roundoff/](paper/roundoff/))
- **Adjoint Scaling Tests**: Validate mixed precision scaling ([paper/adjoint_scaling/](paper/adjoint_scaling/))

### Running Experiments

Each experiment directory contains:

- `run_test.sh`: Quick validation test (reduced iterations)
- `run_experiment.sh`: Full experiment for paper results
- `process_results.sh`: Generate figures and tables from results

```bash
# Example: Run CNF experiment test
cd paper/cnf
./run_test.sh

# Process all experiment results
cd paper
python process_all_results.py
```

For detailed instructions, see [paper/README.md](paper/README.md).

## Testing

Run all core library tests:

```bash
python tests/run_all_tests.py
```

Run specific test suites:

```bash
# Numerical accuracy comparison with torchdiffeq
python tests/core/test_rampde.py

# Gradient quality tests using Taylor expansion
python tests/core/test_backward.py

# Integration and convergence tests
python tests/core/test_odeint.py

# Mixed precision scaling validation
python tests/core/test_adjoint_scaling.py
```

The test suite compares rampde solutions and gradients against torchdiffeq under identical conditions (float32) to ensure numerical consistency. Taylor expansion tests validate gradient approximation quality.

## Acknowledgements 

This work was supported by the Office of Naval Research award N00014-24-1-2221/ P00003 and also in part by NSF award DMS 2038118. The project was supported by the Horizon Europe, MSCA-SE project 101131557 (REMODEL).