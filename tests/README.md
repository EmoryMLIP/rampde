# rampde Test Suite

This folder contains the test suite for the rampde package, organized into core functionality tests and performance regression tests.

## Test Organization

### Core Tests (`core/`)

Core functionality tests that verify numerical correctness, gradient accuracy, and compatibility with torchdiffeq:

- **test_rampde.py**: Compares solutions and gradients of rampde's ODE solver to torchdiffeq for both linear and neural ODEs
- **test_rampde_tuple.py**: Tests tuple-valued ODEs for compatibility between rampde and torchdiffeq
- **test_backward.py**: Verifies the accuracy of backward-mode derivatives (input, weights, time) using Taylor expansion tests
- **test_odeint.py**: Confirms the convergence order of the custom ODE solvers (Euler and RK4) against analytical solutions
- **test_adjoint_scaling.py**: Checks the accuracy of gradients in low-precision time integration, especially with dynamic scaling for float16
- **test_ode_gradients_simple.py**: Simplified gradient correctness tests
- **test_dtype_preservation.py**: Tests that data types are preserved correctly through ODE integration
- **test_speed.py**: Basic speed/performance tests
- **simple_gradient_test.py**: Basic gradient check functionality
- **test_adjoint_scaling/**: Directory containing additional adjoint scaling test resources (PNG files)

### Performance Tests (`performance/`)

Performance regression tests that monitor speed and detect performance degradation:

- **test_performance_regression.py**: Main regression test suite with all solver variants
- **test_otflow_performance.py**: Complex OTFlow performance tests
- **baselines/**: Performance baseline JSON files
- **utils/**: Timing and comparison utilities

See `performance/README.md` for detailed performance testing documentation.

## Running Tests

### Run All Tests

From the repository root:

```bash
# Run core library tests only
python tests/run_all_tests.py

# Run core tests with performance tests included
python tests/run_all_tests.py --include-performance
```

### Run Specific Test Files

```bash
# Run specific core test
python tests/core/test_rampde.py

# Run specific performance test
python tests/performance/test_performance_regression.py

# Run with verbose output (remove RAMPDE_TEST_QUIET env var)
RAMPDE_TEST_QUIET=0 python tests/core/test_rampde.py
```

## Notes

- Some tests require a CUDA-capable GPU
- Tests marked with `test_rampde.py` and `test_rampde_tuple.py` require `torchdiffeq` to be installed; they will be skipped if it is not available
- Test output will summarize any failures or errors at the end
- Performance tests are excluded by default to keep CI fast; use `--include-performance` to run them
