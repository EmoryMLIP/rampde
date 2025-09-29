# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rampde is a PyTorch-compatible library for high-performance, mixed-precision solvers for Neural Ordinary Differential Equations (ODEs). It provides seamless integration with PyTorch's autocast and torchdiffeq package, supporting both forward and backward computations with customizable precision.

## Core Architecture

### Main Components

- **rampde/odeint.py**: Main entry point providing the `odeint` function with API compatibility to torchdiffeq
- **rampde/fixed_grid.py**: Core solver implementations (Euler, RK4, FixedGridODESolver)
- **rampde/loss_scalers.py**: Mixed precision scaling components (DynamicScaler)
- **rampde/fixed_grid_fast.py**: Optimized fixed grid solver implementation

### Key Features

- Drop-in replacement for torchdiffeq with mixed precision support
- Automatic precision handling via PyTorch's autocast
- Dynamic loss scaling for stable gradient computation
- Support for both tensor and tuple inputs (following torchdiffeq conventions)
- Fixed grid solvers optimized for performance

## Installation

### Development Installation

For development work, install the package in editable mode:

```bash
# Activate your conda environment
conda activate torch26

# Install in editable mode with development dependencies (includes torchdiffeq)
pip install -e ".[dev]"

# Or install just the core package (without torchdiffeq)
pip install -e .
```

### Production Installation

Once published to PyPI, the package can be installed via:

```bash
pip install rampde
```

### Optional Dependencies

The package now makes torchdiffeq an optional dependency. Install with optional dependencies for different use cases:

```bash
# For benchmarking and comparison with torchdiffeq
pip install rampde[benchmarks]

# For testing (includes torchdiffeq for comparison tests)
pip install rampde[testing]

# For development (includes all dependencies)
pip install rampde[dev]
```

## Development Commands

### Testing

```bash
# Run all core library tests
python tests/run_all_tests.py

# Run core tests only
python tests/run_all_tests.py

# Run performance tests
python tests/run_all_tests.py --include-performance

# Run specific core test file
python tests/core/test_rampde.py

# Run tests with verbose output (remove TORCHMPNODE_TEST_QUIET env var)
TORCHMPNODE_TEST_QUIET=0 python tests/core/test_rampde.py

# Run gradient quality tests
python tests/core/test_backward.py
python tests/core/test_ode_gradients_simple.py

# Run integration and scaling tests
python tests/core/test_odeint.py
python tests/core/test_adjoint_scaling.py

# Run experiment-specific tests
python experiments/stl10/tests/test_ode_stl10.py
python experiments/stl10/tests/test_ode_stl10_simple.py
```

### Running Examples

```bash
# Basic ODE demo
python examples/ode_demo.py

# Linear ODE example
python examples/modeleq_demo.py

# MNIST neural ODE classifier
python examples/ode_mnist.py

# Continuous normalizing flow
python examples/cnf8g.py

# Optimal transport flow
python examples/otflow.py
```

### Running Demo Scripts

```bash
# Taylor expansion sensitivity analysis
python demos/demo_taylor.py

# Autocast behavior demonstration
python demos/demo_diffeq_autocast.py

# Dynamic scaling benefits
python demos/demo_scaler.py

# Performance benchmarking
python demos/demo_speedup.py
```

### Experiment Scripts

The `experiments/` directory contains comprehensive benchmarking scripts:

```bash
# MNIST experiments
bash experiments/mnist/run_mnist.sh

# STL10 experiments  
bash experiments/stl10/run_stl10.sh

# CNF experiments
bash experiments/cnf/run_cnf.sh

# Large-scale optimal transport flow experiments
bash experiments/otflowlarge/run_largeot.sh

# OT flow experiments
bash experiments/otflow/run_otflow.sh
```

## Code Structure Patterns

### ODE Function Definition

ODE functions should inherit from `torch.nn.Module` and implement `forward(self, t, y)`:

```python
class ODEFunc(nn.Module):
    def forward(self, t, y):
        return self.net(y)  # or any function of t and y
```

### Using rampde

Replace torchdiffeq imports:
```python
# from torchdiffeq import odeint
from rampde import odeint
```

The API is identical, with additional mixed precision support via autocast.

### Mixed Precision Usage

rampde automatically detects autocast context and applies appropriate precision:

```python
with torch.autocast(device_type='cuda'):
    solution = odeint(func, y0, t, method='rk4')
```

## Test Structure

The test suite is organized into logical sections:

### Core Library Tests (`tests/core/`)
- **test_rampde.py**: Numerical accuracy comparison with torchdiffeq
- **test_rampde_tuple.py**: Tuple-valued ODE compatibility tests
- **test_backward.py**: Gradient quality tests using Taylor expansion
- **test_odeint.py**: Integration tests for the main odeint function
- **test_adjoint_scaling.py**: Mixed precision scaling validation
- **test_speed.py**: Performance benchmarking tests
- **test_ode_gradients_simple.py**: Simplified gradient correctness tests
- **simple_gradient_test.py**: Basic gradient check functionality

### Performance Tests (`tests/performance/`)
- **test_performance_regression.py**: Performance regression detection
- **test_otflow_performance.py**: Complex ODE performance benchmarking
- **utils/**: Timing and comparison utilities

### Experiment-Specific Tests (`experiments/[experiment]/tests/`)
- **experiments/stl10/tests/**: STL10 experiment tests that depend on local experiment code

Tests compare solutions and gradients between rampde and torchdiffeq under identical conditions (float32) to ensure numerical consistency.

## Dependencies

### Core Dependencies
- torch >= 2.0 (for autocast support)
- numpy

### Optional Dependencies
- torchdiffeq (for comparison testing and benchmarking - install via `pip install rampde[testing]` or `pip install rampde[benchmarks]`)
- matplotlib (for visualization examples - install via `pip install rampde[examples]`)
- pytest, pytest-cov (for testing - install via `pip install rampde[testing]`)

Additional experiment dependencies are in `experiments/otflowlarge/requirements.txt`.

## Performance Considerations

- Use autocast for mixed precision benefits
- Fixed grid solvers (RK4, Euler) are optimized for performance
- Dynamic scaling helps maintain gradient stability in mixed precision
- Batch processing and tensor operations are GPU-optimized

## Working Directory Context

When working in specific experiment directories like `experiments/otflowlarge/`, note that:
- Local requirements may differ (see `requirements.txt` in experiment folders)
- SLURM batch scripts (`.sbatch` files) are used for HPC environments
- Result directories are created locally for experiment outputs
- Experiment-specific shell scripts handle parameter sweeps and job submission

## Development Environment Reminders

- Always activate torch26 before using python
- Conda activation commands:
  * `conda activate torch26` to activate the specific environment

## HPC Utilities

- To start GPU session run "srun --gres=gpu:1 --time=08:00:00 --pty /bin/bash"