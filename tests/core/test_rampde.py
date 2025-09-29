"""
Unit tests for comparing the solutions and gradients of ODE solvers.

This test suite verifies that the custom ODE solver (`mpodeint` from `torchmpnode`) 
produces results consistent with `torchdiffeq.odeint`. The tests evaluate both 
the final solution and gradients using different ODE function types:

1. **Linear ODE Function**: dx/dt = -(Aᵀ (A x)) * 1e-2
2. **Neural Network ODE Function**: dx/dt = NN(x³), where NN is a small MLP.

For each ODE function:
- Solutions are computed using both solvers.
- Errors between the solutions are printed and checked.
- Gradients of the learned parameters are compared.
- Tests run on both CPU and CUDA (if available).
- Tests run using float32 (since gradients are expected to differ between solvers).

"""


import unittest
import torch
import torch.nn as nn
import os, sys, copy
from torchmpnode import odeint as mpodeint

# Try to import torchdiffeq, skip all tests if not available
try:
    from torchdiffeq import odeint as torch_odeint
    _TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    _TORCHDIFFEQ_AVAILABLE = False



@unittest.skipUnless(_TORCHDIFFEQ_AVAILABLE, "torchdiffeq is not available")
class TestODEintEquivalence(unittest.TestCase):

    def setUp(self):
        # Define the linear ODE function: dx/dt = -(Aᵀ (A x))*1e-2
        class LinearODEFunc(torch.nn.Module):
            def __init__(self, d=10):
                super(LinearODEFunc, self).__init__()
                self.A = torch.nn.Parameter(torch.randn(d, d))
            def forward(self, t, x):
                return -(self.A.transpose(0, 1) @ (self.A @ x)) * 1e-2
        self.LinearODEFunc = LinearODEFunc  # save the class (a factory)

        # Define your new neural ODE function.
        class ODEFunc(nn.Module):
            def __init__(self):
                super(ODEFunc, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 50),
                    nn.Tanh(),
                    nn.Linear(50, 2),
                )
                # Initialize weights and biases.
                for m in self.net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                        nn.init.constant_(m.bias, val=0)
            def forward(self, t, y):
                return self.net(y**3)
        self.ODEFunc = ODEFunc  # save the class (a factory)

    def _run_test(self, func_factory, x0, t, device, method='rk4'):
        """
        This helper creates two instances (with identical initial parameters)
        using the passed factory, and then runs both torchdiffeq and mpodeint.
        It compares both the final solution and the gradients.
        """
        # Create two identical instances.
        f1 = func_factory()
        f2 = func_factory()

        # Copy the parameters from f1 to f2.
        f2.load_state_dict(copy.deepcopy(f1.state_dict()))

        f1.to(device)
        f2.to(device)
        x0 = x0.to(device)
        t = t.to(device)

        # --- Solve with torchdiffeq using f1 ---
        torch_solution = torch_odeint(f1, x0, t, method=method)
        # Compute a simple scalar from the final state and backpropagate.
        def loss(x):
            return torch.mean(torch.abs(x))
            # return torch.norm(x)
        
        loss(torch_solution).backward()
        grad_params = [p.grad for p in f1.parameters() if p.requires_grad]

        # --- Solve with mpodeint using f2 ---
        my_solution = mpodeint(f2, x0, t, method=method)
        loss(my_solution).backward()
        my_grad_params = [p.grad for p in f2.parameters() if p.requires_grad]

        quiet = os.environ.get("TORCHMPNODE_TEST_QUIET", "0") == "1"
        if not quiet:
            print("Torchdiffeq grads:")
            for name, param in f1.named_parameters():
                print("Parameter name: ", name, "grad: ",torch.norm(param.grad))        
            print("mpodeint grads:")
            for name, param in f2.named_parameters():
                print("Parameter name: ", name, "grad: ",torch.norm(param.grad))        
            print("Torchdiffeq final state:", torch_solution[-1])
            print("mpodeint final state:", my_solution[-1])
            print("Solution absolute error:", torch.norm(torch_solution - my_solution).item())
            print("Solution relative error:", torch.norm((torch_solution - my_solution) / torch.norm(torch_solution)).item())
            for g1, g2 in zip(grad_params, my_grad_params):
                print("Gradient absolute error:", torch.norm(g1 - g2).item())
                print("Gradient relative error:", torch.norm((g1 - g2) / torch.norm(g1)).item())

        # Compare the final solutions.
        self.assertTrue(torch.allclose(my_solution, torch_solution, rtol=1e-5, atol=1e-5),
                        "The solutions from torchmpnode and torchdiffeq differ more than expected.")
        # Compare gradients for each parameter.
        for g1, g2 in zip(grad_params, my_grad_params):
            self.assertTrue(torch.allclose(g1, g2, rtol=1e-5, atol=1e-5),
                        "The gradients from torchmpnode and torchdiffeq differ more than expected.")

    def test_linear_odefunc_on_cpu(self):
        print("\nRunning test_linear_odefunc_on_cpu\n")
        device = torch.device('cpu')
        d = 10
        x0 = torch.ones(d)
        t = torch.linspace(0, 10, 100)
        # Pass a lambda that creates a new LinearODEFunc instance.
        self._run_test(lambda: self.LinearODEFunc(d), x0, t, device, method='euler')

    def test_neural_odefunc_on_cpu(self):
        print("\nRunning test_neural_odefunc_on_cpu\n")
        device = torch.device('cpu')
        # ODEFunc expects an input of size 2.
        x0 = torch.ones(2)
        t = torch.linspace(0., 25., 1000).to(device)
            
        self._run_test(lambda: self.ODEFunc(), x0, t, device, method='euler')

    def test_linear_odefunc_on_cuda(self):
        print("\nRunning test_linear_odefunc_on_cuda\n")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            d = 10
            x0 = torch.ones(d)
            t = torch.linspace(0, 10, 100)
            self._run_test(lambda: self.LinearODEFunc(d), x0, t, device, method='euler')
        else:
            self.skipTest("CUDA not available")

    def test_neural_odefunc_on_cuda(self):
        print("\nRunning test_neural_odefunc_on_cuda\n")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x0 = torch.ones(2)
            # t = torch.linspace(0, 10, 100)
            t = torch.linspace(0., 25., 1000)
            self._run_test(lambda: self.ODEFunc(), x0, t, device, method='euler')
        else:
            self.skipTest("CUDA not available")

if __name__ == '__main__':
    unittest.main()