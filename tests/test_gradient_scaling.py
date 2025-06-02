import unittest
import math
import torch
import torch.nn as nn
import sys, os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint, NoScaler, DynamicScaler
from torch.amp import autocast

torch.set_default_dtype(torch.float32)

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
def solve_analytically(model, y0, t):
    A = model.A.detach().clone().to(torch.float64).requires_grad_(True)
    y0 = y0.detach().clone().to(torch.float64).requires_grad_(True)
    # Compute the matrix exponential of A * t.
    exp_At = torch.matrix_exp(A * t.max())
    # Compute the solution at t.
    y_t = exp_At @ y0
    # Compute the gradients.
    grad_y0, grad_A = torch.autograd.grad(y_t, (y0,A), grad_outputs=torch.ones_like(y_t), retain_graph=True)
    return y_t, grad_y0, grad_A

def solve_ode(model, y0, t, method='rk4', working_dtype=torch.float32, scaler = DynamicScaler):
    with autocast(device_type='cuda', dtype=working_dtype):        
        return odeint(model, y0, t, method=method, loss_scaler=scaler(working_dtype))


# Helper function to compute gradients with respect to y0 and model.A.
def compute_gradients(model, y0, t, method,  working_dtype=torch.float32, scaler=DynamicScaler):
    # Ensure y0 is a leaf tensor requiring gradient.
    y0 = y0.detach().clone().requires_grad_(True)
    # For simplicity, we assume t does not require grad.
    A_temp = model.A.data.clone()
    model.A.data = model.A.to(torch.float32)
    y0 = y0.to(torch.float32)
    t = t.to(torch.float32)
    # Use autocast for the forward pass.
    with autocast(device_type='cuda', dtype=working_dtype):
        y0.grad = None
        model.A.grad = None
        sol = solve_ode(model, y0, t, method=method, working_dtype=working_dtype,scaler = scaler)
        loss = sol[-1].sum()
        loss.backward()
    # Restore the original A parameter.
    grad_y0 = y0.grad.detach().clone()
    grad_A = model.A.grad.detach().clone()
    
    model.A.data = A_temp
    return sol, grad_y0, grad_A

class TestGradientPrecisionComparison(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            print("GPU not available. Skipping tests.")
            self.skipTest("GPU required for these tests.")
        self.device = torch.device("cuda:0")
        self.dim = 2  # Small state dimension for clarity.
        # Create a time grid from 0 to 1.
        self.t = torch.linspace(0., 1.0, 40, device=self.device)
        # Initial state.
        self.y0 = torch.randn(self.dim, device=self.device)
        # Create the ODE model.
        self.model = LinearODE(self.dim).to(self.device)
        # Save a copy of the original parameter for later restoration if needed.
        self.original_A = self.model.A.detach().clone()

    def test_precision_vs_analytic(self):
        # Compute analytic solution and gradients in float64
        
        y_T_analytic, grad_y0_analytic, grad_A_analytic = solve_analytically(self.model, self.y0, self.t)
        # Run numerical solution with NoScaler in float64
        
        # Prepare to compute relative errors for different dtypes and scalers
        results = []

        scalers_str = ["NoScaler", "DynamicScaler"]
        for working_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for (scaler, name_str) in zip([NoScaler, DynamicScaler], scalers_str):
                self.model.A.data.copy_(self.original_A)
                # sol = solve_ode(self.model, self.y0.to(working_dtype), self.t.to(working_dtype), use_autocast=False, working_dtype=working_dtype)[-1]
                
                try:
                    sol, grad_y0_numeric, grad_A_numeric = compute_gradients(self.model, self.y0, self.t, method='rk4', working_dtype=working_dtype, scaler=scaler)
                    y_T_numeric = sol[-1].detach()
    
                    rel_err_state = torch.linalg.norm(y_T_numeric - y_T_analytic) / torch.linalg.norm(y_T_analytic)
                    rel_err_grad_y0 = torch.linalg.norm(grad_y0_numeric.detach() - grad_y0_analytic) / torch.linalg.norm(grad_y0_analytic)
                    rel_err_grad_A = torch.linalg.norm(grad_A_numeric.detach() - grad_A_analytic) / torch.linalg.norm(grad_A_analytic)
    
                    results.append((
                        str(working_dtype),
                        name_str,
                        f"{rel_err_state:.8e}",
                        f"{rel_err_grad_y0:.8e}",
                        f"{rel_err_grad_A:.8e}"
                    ))
                except RuntimeError as e:
                    error_message = f"RuntimeError: {str(e)}"
                    error_message = 'fail'
                    results.append((
                        str(working_dtype),
                        name_str,
                        error_message,
                        error_message,
                        error_message
                    ))
                
        # Print results in a markdown-like table format
        print("| Working dtype | Scaler         | Rel Err State | Rel Err Grad y0 | Rel Err Grad A |")
        print("|---------------|----------------|---------------|------------------|----------------|")
        for dtype, scaler, err_state, err_grad_y0, err_grad_A in results:
            print(f"| {dtype}         | {scaler  }\t | {err_state}     | {err_grad_y0}         | {err_grad_A}     |")

    def test_positive_definite_system(self):
        """Test with A = original_A^T @ original_A (positive definite)."""
        print("Testing positive definite system")
        A = 0.6*torch.randn_like(self.original_A)
        self.model.A.data.copy_(A.T @ A)
        # Re-run precision vs analytic for this A
        self.original_A = A.T @ A
        self.test_precision_vs_analytic()

    def test_negative_definite_system(self):
        """Test with A = - (original_A^T @ original_A) (negative definite)."""
        print("Testing negative definite system")
        A = .15*torch.randn_like(self.original_A)
        self.model.A.data.copy_(-A.T @ A)
        # Re-run precision vs analytic for this A
        self.original_A =- A.T @ A
        # Re-run precision vs analytic for this A
        self.test_precision_vs_analytic()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    unittest.main(argv=[sys.argv[0]] + (['-v'] if args.verbose else []))