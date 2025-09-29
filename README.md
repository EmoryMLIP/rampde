# rampde
rampde is a PyTorch-compatible library designed to provide high-performance, mixed-precision solvers for Neural Ordinary Differential Equations (ODEs). The package integrates seamlessly with PyTorch’s ecosystem, allowing users to replace standard solvers with mixed-precision alternatives for faster computation and reduced memory usage.

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

Note: `torchdiffeq` is now an optional dependency. The core rampde functionality works without it. Install `torchdiffeq` separately if needed for comparisons.
  
## Experiments

1. A linear ODE example to demonstrate the relative error of numerical solution and sensitivities of loss with respect to input and weights.

```
python modeleq_demo.py
```

2. Learning the dynamics of a nonlinear ODE

```
python ode_demo.py
```

3. Learning a classifier for MNIST dataset

```
python ode_mnist.py
```

4. Continuous normalizing flow and OT flow

```
python cnf8g.py
```

```
python otflow.py
```

## Demo

We check the sensitivity of loss with respect to input perturbations in 
```
python demos/demo_taylor.py
```

We can see explicitly the behavior of torchdiffeq under Autocast in
```
python demos/demo_diffeq_autocast.py
```

We demonstrate the benefits of dynamic (adjoint) scaling in 
```
python demos/demo_scaler.py
```

## Tests

Use tests/test_rampde.py, to check numerical solution and gradients under the same high precision (f32). 
Use tests/test_backward_input.py, test_backward_weights.py and test_backward_time.py to test the quality of the gradient approximation based on Taylor expansion. 
In tests/test_rampde.py, we see that torchdiffeq and rampde performs consistently in terms of numerical solution and gradients under the same high precision (f32). However, tests/test_backward_input.py, test_backward_weights.py and test_backward_time.py show that rampde yields better gradient approximations. Further tests on rampde and gradient scaling can be found in tests/test_odeint.py and tests/test_gradient_scaling.py.





