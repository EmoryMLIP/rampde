# ODE STL10 Gradient Testing

This directory contains comprehensive gradient tests for the ODE STL10 model components. These tests are designed to verify the correctness of gradient computations, especially for the piecewise constant weight interpolation system.

## Test Files

### 1. `test_ode_stl10.py` - Full Model Gradient Tests
**Purpose**: Test gradients for the complete ODE STL10 model components.

**Features**:
- Tests `ODEFunc`, `ODEBlock`, and `MPNODE_STL10` classes
- Uses double precision for accurate gradient checking
- Configurable precision testing (float64, float32)
- Handles missing dependencies gracefully

**Usage**:
```bash
# Run all tests
python test_ode_stl10.py

# Run specific test for debugging
python test_ode_stl10.py debug
```

### 2. `test_ode_gradients_simple.py` - Simplified Gradient Tests
**Purpose**: Standalone gradient tests that don't depend on the full ODE STL10 model.

**Features**:
- Self-contained test with simplified ODE components
- Tests gradient accuracy across different precisions
- Numerical vs analytical gradient comparison
- Precision robustness testing (float64, float32, float16)

**Usage**:
```bash
python test_ode_gradients_simple.py
```

### 3. `test_piecewise_constant.py` - Piecewise Constant Weight Tests
**Purpose**: Specifically test the piecewise constant weight interpolation mechanism.

**Features**:
- Tests time-dependent weight indexing
- Verifies gradient flow through piecewise constant system
- Tests boundary conditions and edge cases
- Tests different numbers of time intervals

**Usage**:
```bash
python test_piecewise_constant.py
```

### 4. `simple_gradient_test.py` - Basic Gradient Check
**Purpose**: Simple sanity check for gradient computation.

**Features**:
- Basic quadratic function gradient test
- Numerical vs analytical comparison
- Quick verification that gradient checking works

**Usage**:
```bash
python simple_gradient_test.py
```

## How Gradient Checking Works

The gradient tests use **finite difference approximation** to compare numerical and analytical gradients:

1. **Analytical Gradient**: Computed using PyTorch's autograd
2. **Numerical Gradient**: Computed using finite differences:
   ```
   ∇f(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
   ```
3. **Comparison**: Check if `|analytical - numerical| < tolerance`

## Test Parameters

### Precision Settings
- **Double Precision (float64)**: Most accurate, used for reference
- **Single Precision (float32)**: Standard training precision
- **Half Precision (float16)**: Low precision, may have gradient issues

### Tolerance Settings
```python
# Double precision
rtol = 1e-6, atol = 1e-8

# Single precision  
rtol = 1e-4, atol = 1e-6

# Half precision
rtol = 1e-2, atol = 1e-4
```

## Debugging Float16 Issues

The gradient tests can help identify float16 problems:

1. **Run precision comparison**:
   ```bash
   python test_ode_gradients_simple.py
   ```

2. **Check for**:
   - Gradient underflow (gradients → 0)
   - Gradient overflow (gradients → ∞)
   - Poor gradient accuracy
   - NaN/infinite values

3. **Common float16 issues**:
   - Loss scaling needed for stable gradients
   - Parameter values too large/small
   - Aggressive parameter clamping
   - Accumulation of rounding errors

## Expected Results

### Successful Test Output
```
=== Testing SimpleODEFunc Gradients ===
Using device: cuda:0, dtype: torch.float64
Checking gradients for input_x...
  input_x: Max error = 1.23e-08, Failed elements = 0/20
Checking gradients for time_t...
  time_t: Max error = 4.56e-09, Failed elements = 0/1
✓ SimpleODEFunc gradient check PASSED
```

### Failed Test Output
```
Checking gradients for conv.weight...
  Element 42: Numerical=1.23e-04, Analytical=1.56e-04, Error=3.30e-05
  conv.weight: Max error = 3.30e-05, Failed elements = 5/20
✗ SimpleODEFunc gradient check FAILED
```

## Troubleshooting

### Import Errors
If you get import errors:
```bash
# Make sure you're in the right environment
conda activate torch26

# Check if rampde is available
python -c "import rampde; print('OK')"

# Run standalone tests that don't need rampde
python test_ode_gradients_simple.py
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU testing if needed
CUDA_VISIBLE_DEVICES="" python test_ode_gradients_simple.py
```

### Memory Issues
- Reduce batch sizes in test parameters
- Use smaller models (fewer channels)
- Test fewer parameters at once

## Integration with Main Code

These tests complement the main training loop's NaN/infinite detection:

1. **Development**: Use these tests to verify gradient correctness
2. **Debugging**: Run when training fails or produces NaN
3. **Precision**: Verify float16 compatibility before training
4. **Validation**: Ensure model changes don't break gradients

## Recommended Testing Workflow

1. **Start simple**: Run `simple_gradient_test.py`
2. **Test components**: Run `test_piecewise_constant.py`
3. **Full testing**: Run `test_ode_gradients_simple.py`
4. **Integration**: Run `test_ode_stl10.py` (if dependencies available)

This progressive approach helps isolate issues and ensures the gradient computation pipeline is working correctly.
