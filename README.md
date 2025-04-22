# torchmpnode
torchmpnode is a PyTorch-compatible library designed to provide high-performance, mixed-precision solvers for Neural Ordinary Differential Equations (ODEs). The package integrates seamlessly with PyTorch’s ecosystem, allowing users to replace standard solvers with mixed-precision alternatives for faster computation and reduced memory usage.

Key features include:
<ul>
	<li>Easy API compatibility with torchdiffeq.</li>
	<li>Support for both forward and backward computations with customizable precision.</li>
	<li>Benchmark tools for performance and memory profiling.</li>
	<li>Examples and tests for various ODE problems, including high-dimensional systems and neural ODE models.</li>
</ul>

## Experiments

1. A linear ODE example to demonstrate the relative error of numerical solution and sensitivities of loss with repsect to input and weights.

```
python modeleq_demo.py
```

Use tests/test_backward_input.py, test_backward_weights.py and test_backward_time.py to test the quality of the gradient approximation based on Taylor expansion.


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

## Tests

1. Torchdiffeq behavior under Autocast
```
python tests/test_autocast_odeint.py
```


