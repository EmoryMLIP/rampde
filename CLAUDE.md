# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rampde is a PyTorch-compatible library for high-performance, mixed-precision solvers for Neural Ordinary Differential Equations (ODEs). It provides seamless integration with PyTorch's autocast and torchdiffeq package, supporting both forward and backward computations with customizable precision.

## Core Architecture

### Main Components

- **rampde/odeint.py**: Main entry point providing the `odeint` function with API compatibility to torchdiffeq
- **rampde/increment.py**: Increment functions (Euler, RK4) for explicit ODE schemes
- **rampde/fixed_grid_base.py**: Base class for fixed grid ODE solvers
- **rampde/fixed_grid_dynamic.py**: Dynamic scaling solver variant (with DynamicScaler for float16)
- **rampde/fixed_grid_unscaled.py**: Optimal performance solver variant (for float32, bfloat16)
- **rampde/fixed_grid_unscaled_safe.py**: Exception handling solver variant (float16 without DynamicScaler)
- **rampde/loss_scalers.py**: Mixed precision scaling components (DynamicScaler)
- **rampde/utils.py**: Utility functions for overflow detection

### Key Features

- Drop-in replacement for torchdiffeq with mixed precision support
- Automatic precision handling via PyTorch's autocast
- Dynamic loss scaling for stable gradient computation
- Support for both tensor and tuple-valued ODE functions (following torchdiffeq conventions)
- Fixed grid solvers optimized for performance

## Installation

### Development Installation

For development work, install the package in editable mode:

```bash
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

The package makes torchdiffeq an optional dependency for testing and evaluation. Install with optional dependencies for different use cases:

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
# Run core library tests
python tests/run_all_tests.py

# Run core tests with performance tests included
python tests/run_all_tests.py --include-performance

# Run specific core test file
python tests/core/test_rampde.py

# Run tests with verbose output (remove RAMPDE_TEST_QUIET env var)
RAMPDE_TEST_QUIET=0 python tests/core/test_rampde.py

# Run gradient quality tests
python tests/core/test_backward.py
python tests/core/test_ode_gradients_simple.py

# Run integration and scaling tests
python tests/core/test_odeint.py
python tests/core/test_adjoint_scaling.py

```

### Paper Reproducibility Scripts

The `paper/` directory contains all experiments from the rampde paper:

```bash
# View available experiments
ls paper/

# Each experiment has:
# - run_test.sh: Quick validation test (2-3 epochs/iterations)
# - run_experiment.sh: Full experiment (for paper results)
# - process_results.sh: Generate figures and tables

# Example: Run CNF experiment test
cd paper/cnf
./run_test.sh

# Example: Run STL10 experiment test
cd paper/stl10
./run_test.sh

# Process all results to generate paper figures/tables
cd paper
python process_all_results.py
```

For detailed experiment instructions, see `paper/README.md`.

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

Tests compare solutions and gradients between rampde and torchdiffeq under identical conditions (float32) to ensure numerical consistency.

## Dependencies

### Core Dependencies

- torch >= 2.0 (for autocast support)
- numpy

### Optional Dependencies

- torchdiffeq (for comparison testing and benchmarking - install via `pip install rampde[testing]` or `pip install rampde[benchmarks]`)
- matplotlib (for visualization in paper scripts - install via `pip install matplotlib`)
- pytest, pytest-cov (for testing - install via `pip install rampde[testing]`)

Additional paper experiment dependencies are in `requirements-paper.txt`.

## Performance Considerations

- Use autocast for mixed precision benefits
- Fixed grid solvers (RK4, Euler) are optimized for performance
- Dynamic scaling helps maintain gradient stability in mixed precision
- Batch processing and tensor operations are GPU-optimized

## Working Directory Context

When working in the `paper/` directory experiments (e.g., `paper/otflowlarge/`), note that:

- Paper experiments require dependencies from `requirements-paper.txt`
- SLURM batch scripts (`.sbatch` files) are used for HPC environments
- Result directories (`raw_data/`, `outputs/`) are created locally for experiment outputs
- Experiment-specific shell scripts handle parameter sweeps and job submission
- See `paper/README.md` for detailed experiment documentation

## Development Environment Reminders

- Always activate torch28 before using python
- Conda activation commands:
  - `conda activate torch28` to activate the specific environment

## HPC Utilities

- To start GPU session run "srun --gres=gpu:1 --time=08:00:00 --pty /bin/bash"

### Repository Structure

- Paper reproducibility scripts are in `paper/` directory
- Core library is in `rampde/`, tests are in `tests/core/` and `tests/performance/`
- No `examples/`, `demos/`, or `experiments/` directories at root level
- See `paper/README.md` for information about running paper experiments
