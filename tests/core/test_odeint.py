"""

This test verifies the convergence order of the custom ODE solvers (Euler and RK4) implemented in torchmpnode. 
It uses a linear ODE with a known analytical solution and checks that the numerical solution converges at the 
expected rate as the number of time steps increases. The test passes if the observed order of convergence matches 
the theoretical order for each solver in at least 4 out of 9 step doublings.

Key points:
- Uses a linear ODE with an analytical solution for accuracy reference.
- Tests both Euler and RK4 methods.
- Checks that the error decreases at the expected rate as the number of steps increases.
- Passes if the observed order is close to the theoretical order for most refinements.
"""
import torch
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint
from torchmpnode import Euler, RK4, FixedGridODESolverUnscaled
from torch.amp import autocast

class TestFixedGridODESolver(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.float64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A = torch.randn(2, 2, dtype=self.dtype, device=self.device)
        self.A = - self.A @ self.A.T *0.05
        class FuncModule(torch.nn.Module):
            def __init__(self, A):
                super(FuncModule, self).__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False,device=A.device)
                self.linear.weight = torch.nn.Parameter(A)

            def forward(self, t, y):
                return self.linear(y)

        self.func = FuncModule(self.A)
        self.y0 = torch.tensor([1.0, 0.0], dtype=self.dtype, device=self.device)
        self.T = 20
        self.num_steps = 100
        self.t = torch.linspace(0, self.T, self.num_steps + 1, dtype=self.dtype, device=self.device)

    def analytical_solution(self, t):
        exp_At = torch.matrix_exp(self.A * t)
        return torch.matmul(exp_At, self.y0)
    
    def analytical_derivative(self,t):
        exp_At = torch.matrix_exp(self.A * t)
        return t*torch.matmul(exp_At, self.y0), exp_At
    
    def test_convergence(self):
        solvers = [Euler(), RK4()]
        quiet = os.environ.get("TORCHMPNODE_TEST_QUIET", "0") == "1"
        for solver in solvers:
            with self.subTest(solver=solver):
                order = solver.order
                pass_count = 0
                previous_error = None
                num_steps = 4
                y_analytical = self.analytical_solution(self.T)
                for _ in range(10):
                    self.num_steps = num_steps
                    self.t = torch.linspace(0, self.T, self.num_steps + 1, dtype=self.dtype, device=self.device)
                    with autocast(device_type='cpu',dtype= self.dtype):
                        yt = odeint(self.func, self.y0, self.t, method=solver.name)
                    error = torch.norm(yt[-1] - y_analytical, dim=-1).max().item()
                    if previous_error is not None:
                        observed_order = np.log2(previous_error / error)
                        if observed_order > order-0.5:
                            pass_count += 1
                        if not quiet:
                            print(f"Steps: {self.num_steps}, Error: {error:.2e}, Observed order: {observed_order:.2e}")
                    else:
                        if not quiet:
                            print(f"Steps: {self.num_steps}, Error: {error:.2e}")
                    previous_error = error
                    num_steps *= 2
                self.assertTrue(pass_count >= 4, f"Convergence order for {solver.name} did not meet expectations.")

    
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on {device}")
    unittest.main()