# Performance Regression Tests

This directory contains performance regression tests for rampde to monitor performance against torchdiffeq and detect regressions in the three-variant architecture.

## Test Structure

### Core Performance Tests
- `test_performance_regression.py` - Main regression test suite with all solver variants
- `test_simple_ode_performance.py` - Simple ODE performance tests
- `test_otflow_performance.py` - Complex OTFlow performance tests

### Utilities
- `utils/test_models.py` - Test model definitions
- `utils/timing_utils.py` - Timing and measurement utilities
- `utils/comparison_utils.py` - Comparison against torchdiffeq

### Baselines
- `baselines/simple_ode_baseline.json` - Expected performance baselines for simple ODEs
- `baselines/otflow_baseline.json` - Expected performance baselines for OTFlow

## Running Performance Tests

### Basic Usage
```bash
# Run all performance tests
python -m pytest tests/performance/

# Run specific test
python tests/performance/test_performance_regression.py

# Run with verbose output
python tests/performance/test_performance_regression.py -v
```

### Integration with Main Test Suite
```bash
# Run all tests including performance
python tests/run_all_tests.py --include-performance

# Run only performance tests
python tests/run_all_tests.py --performance-only
```

## Test Configurations

### Solver Variants Tested
- **Unscaled**: Optimal performance for float32/bfloat16
- **Dynamic**: Dynamic scaling for DynamicScaler
- **UnscaledSafe**: Exception handling for float16

### Precision Types Tested
- float32 (with None and NoScaler)
- bfloat16 (with None and NoScaler)
- float16 (with None, NoScaler, and DynamicScaler)

### Performance Metrics
- Mean execution time with standard deviation
- Speedup relative to slowest configuration
- Memory usage tracking
- Solver selection verification
- Comparison against torchdiffeq baseline

## Performance Baselines

The regression tests compare against established baselines to detect performance regressions:

### Simple ODE Baselines
- Unscaled variants: ~0.04s
- UnscaledSafe variants: ~0.047s  
- Dynamic variants: ~0.067s

### OTFlow Baselines
- Unscaled variants: ~0.186s
- UnscaledSafe variants: ~0.207s
- Dynamic variants: ~0.207s

### Regression Thresholds
- **Warning**: >5% slower than baseline
- **Error**: >10% slower than baseline
- **Critical**: >20% slower than baseline

## Adding New Performance Tests

### For New Models
1. Add model definition to `utils/test_models.py`
2. Create test function in appropriate test file
3. Add baseline to `baselines/` directory
4. Document expected performance characteristics

### For New Configurations
1. Add configuration to test matrix
2. Update baseline files
3. Document rationale for new configuration

## Interpreting Results

### Performance Summary Table
Each test outputs a summary table showing:
- Configuration name
- Precision type
- Scaler type
- Solver variant used
- Mean execution time
- Speedup relative to baseline
- Pass/fail status

### Regression Detection
- Tests automatically detect performance regressions
- Warnings are issued for significant slowdowns
- Critical regressions fail the test

### Solver Selection Verification
- Tests verify that the correct solver is selected for each configuration
- Ensures automatic optimization is working correctly

## Historical Context

These tests preserve the research work from the three-variant architecture development:
- Performance improvements of up to 65% over previous implementation
- Systematic ablation study of exception handling approaches
- Comparison against torchdiffeq showing 47% performance improvement

See `tests/research/` for detailed research documentation and historical implementations.