import unittest
import math
import torch
import torch.nn as nn
from torchdiffeq import odeint
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint
from torch.amp import autocast


# Define a simple linear ODE module with a trainable parameter A.
class LinearODE(nn.Module):
    def __init__(self, dim):
        super(LinearODE, self).__init__()
        # A is registered as an nn.Parameter (stored in FP32).
        self.A = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32))
    
    def forward(self, t, y):
        # Cast y to FP32 so that the multiplication with A (in FP32) is performed in FP32.
        return self.A @ y

# Helper function to solve the ODE.
def solve_ode(model, y0, t, method='rk4', use_autocast=False, working_dtype=torch.float32):
    if use_autocast:
        with autocast(device_type='cuda', dtype=working_dtype):
            sol = odeint(model, y0.to(working_dtype), t.to(working_dtype), method=method)
    else:
        sol = odeint(model, y0, t, method=method)
    return sol

# Helper function to compute gradients with respect to y0 and model.A.
def compute_gradients(model, y0, t, method, use_autocast=False, working_dtype=torch.float32):
    # Ensure y0 is a leaf tensor requiring gradient.
    y0 = y0.detach().clone().requires_grad_(True)
    # For simplicity, we assume t does not require grad.
    if use_autocast:
        with autocast(device_type='cuda', dtype=working_dtype):
            sol = solve_ode(model, y0, t, method=method, use_autocast=True, working_dtype=working_dtype)
            loss = sol[-1].sum()
            loss.backward()
    
    else:
        sol = solve_ode(model, y0, t, method=method, use_autocast=False)
        loss = sol[-1].sum()
        loss.backward()

    grad_y0 = y0.grad.detach().clone()
    grad_A = model.A.grad.detach().clone()
    return sol, grad_y0, grad_A

class TestGradientPrecisionComparison(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("GPU required for these tests.")
        self.device = torch.device("cuda:0")
        self.dim = 2  # Small state dimension for clarity.
        # Create a time grid from 0 to 1.
        self.t = torch.linspace(0., 1.0, 10, device=self.device, dtype=torch.float32)
        # Initial state.
        self.y0 = torch.randn(self.dim, device=self.device, dtype=torch.float32)
        # Create the ODE model.
        self.model = LinearODE(self.dim).to(self.device)
        # Save a copy of the original parameter for later restoration if needed.
        self.original_A = self.model.A.detach().clone()

    def _compare_precision(self, working_dtype):
        method = 'rk4'
        # Zero gradients before each run.
        self.model.zero_grad()
        sol_full, grad_y0_full, grad_A_full = compute_gradients(
            self.model, self.y0, self.t, method=method, use_autocast=False,working_dtype=torch.float32)
        # assert sol_full.isfinite().all()
        self.model.zero_grad()
        sol_low, grad_y0_low, grad_A_low = compute_gradients(
            self.model, self.y0, self.t, method=method, use_autocast=True, working_dtype=working_dtype)
        # print(sol_low)
        # assert sol_low.isfinite().all()
        
        # Cast low-precision results back to FP32 for comparison.
        final_full = sol_full[-1]
        final_low = sol_low[-1].to(torch.float32)
        grad_y0_low = grad_y0_low.to(torch.float32)
        grad_A_low = grad_A_low.to(torch.float32)
        
        rel_err_state = torch.norm(final_full - final_low) / (torch.norm(final_full))
        rel_err_grad_y0 = torch.norm(grad_y0_full - grad_y0_low) / (torch.norm(grad_y0_full))
        rel_err_grad_A = torch.norm(grad_A_full - grad_A_low) / (torch.norm(grad_A_full))
        
        print(f"Working dtype: {working_dtype}")
        print("Relative error in terminal state: %1.4e" % rel_err_state.item())
        print("Relative error in grad y0: %1.4e" % rel_err_grad_y0.item())
        print("Relative error in grad A: %1.4e" % rel_err_grad_A.item())
        # We expect differences that reveal inaccuracies when no adaptive scaling is used.
        return rel_err_state.item(), rel_err_grad_y0.item(), rel_err_grad_A.item()

    def test_positive_definite_system(self):
        # For positive definite A, we set A to be symmetric and with positive eigenvalues.
        A_pos = torch.tensor([[1.0, 0.5],
                              [0.5, 1.0]], device=self.device, dtype=torch.float32)
        self.model.A.data.copy_(A_pos)
        print("\nTesting positive definite A")
        for working_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            rel_errs = self._compare_precision(working_dtype)
            # Here you might assert that the relative errors are within a tolerance
            # if you expect low-precision without scaling to give reasonable results.
            # For this example, we do not impose a strict tolerance because we expect discrepancies.
            # self.assertTrue(all(err < 0.1 for err in rel_errs))

    def test_negative_definite_system(self):
        # For negative definite A, set A = -A_pos.
        A_neg = -20*torch.tensor([[1.0, 0.5],
                               [0.5, 1.0]], device=self.device, dtype=torch.float32)
        self.model.A.data.copy_(A_neg)
        print("\nTesting negative definite A")
        for working_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            rel_errs = self._compare_precision(working_dtype)
            # Here, too, one can assert expectations if desired.
            # self.assertTrue(all(err < 0.1 for err in rel_errs))

if __name__ == '__main__':
    unittest.main()