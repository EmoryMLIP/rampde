import torch
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint
from torchmpnode import Euler, RK4, FixedGridODESolver

class TestFixedGridODESolver(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.float64
        self.device = torch.device('cpu')
        self.A = torch.randn(2, 2, dtype=self.dtype, device=self.device)
        self.A = - self.A @ self.A.T 
        class FuncModule(torch.nn.Module):
            def __init__(self, A):
                super(FuncModule, self).__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
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
        for solver in solvers:
            with self.subTest(solver=solver):
                order = solver.order
                pass_count = 0
                previous_error = None
                num_steps = 8
                y_analytical = self.analytical_solution(self.T)
                    
                for _ in range(10):
                    self.num_steps = num_steps
                    self.t = torch.linspace(0, self.T, self.num_steps + 1, dtype=self.dtype, device=self.device)
                    yt = odeint(self.func, self.y0, self.t, method=solver.name, dtype_hi=self.dtype)
                    error = torch.norm(yt[-1] - y_analytical, dim=-1).max().item()
                    if previous_error is not None:
                        observed_order = np.log2(previous_error / error)
                        if observed_order > order-0.5:
                            pass_count += 1
                        print(f"Steps: {self.num_steps}, Error: {error:.2e}, Observed order: {observed_order:.2e}")
                    else: 
                        print(f"Steps: {self.num_steps}, Error: {error:.2e}")
                    previous_error = error
                    num_steps *= 2
                self.assertTrue(pass_count >= 4, f"Convergence order for {solver.name} did not meet expectations.")

    
        
if __name__ == '__main__':
    unittest.main()