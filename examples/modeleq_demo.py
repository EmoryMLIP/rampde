"""
A numerical experiment for mixed-precision time integration of ODEs.

We consider the linear ODE:
    y' = A y,   y(0) = x,  t in [0,1]
with analytical solution:
    y(1) = matrix_exp(A) x.

We then compute the relative error in the solution as well as in the
sensitivities (gradients of the final state with respect to x and A)
when using various ODE solvers in torchdiffeq.
"""

import torch
from torchdiffeq import odeint
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint
from torch.amp import autocast


# --- Define the ODE right-hand side (a linear ODE) ---
class LinearODE(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = torch.nn.Parameter(A)  # A is a torch parameter (or tensor with requires_grad=True)

    def forward(self, t, y, verbose=False):
        # y is assumed to be a vector of shape (D,)
        # Returns A @ y.
        x = self.A.matmul(y)
        if verbose:
            dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
            xt = self.A.to(dtype_low).matmul(y.to(dtype_low))        
            print(f"y = {y.dtype}, A @ y = {x.dtype} self.A = {self.A.dtype}, err: {torch.norm(x.to(torch.float64) - xt.to(torch.float64))}")
        return x

# --- Analytical solution ---
def analytic_solution(A, x, T=1.0):
    """
    Computes the exact solution of y' = A y at time T:
        y(T) = exp(A*T) x.
    """
    expA = torch.matrix_exp(A * T)
    return expA.matmul(x)

# --- Run one experiment ---
def run_experiment(ode,method, dtype_f, dtype_y, *, n_steps=None, rtol=None, atol=None, D=10, T=1.0):
    """
    Runs one experiment:
      - Creates a random initial condition x and parameter A (both with requires_grad)
      - Uses the chosen ODE integrator to compute y(T)
      - Computes the analytic solution y_true = exp(A*T)*x
      - Computes the relative error in the solution and in the sensitivities
        (i.e. the gradients of a scalar loss = sum(y_final) with respect to x and A)
    
    For fixed-step methods ('euler', 'rk4'), you must specify n_steps.
    For dopri5, specify tolerances rtol and atol.
    
    Returns: (rel_error_solution, rel_error_grad_x, rel_error_grad_A)
    """
    # Set the random seed (optional, for reproducibility)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device == torch.device('cuda'), "This script does not support CPU"

    # Create initial condition and parameter A
    x = torch.randn(D, dtype=dtype_y,  device=device, requires_grad=True)
    # For A, we create a D x D matrix.
    A = torch.rand(D, D,  device=device, requires_grad=True)
    # assign A the value of -5 without messing up gradient
    A.data = -5*torch.eye(D,device=device)
    # Make A symmetric
    # Set up the time grid
    if method in ['euler', 'rk4']:
        if n_steps is None:
            n_steps = 100  # default
        # Create a uniform grid from 0 to T (inclusive)
        t = torch.linspace(0.0, T, n_steps + 1, device=device)
        solver_options = {}  # no tolerance options for fixed-step methods
    elif method == 'dopri5':
        # For dopri5, we can simply specify the start and end times.
        t = torch.tensor([0.0, T], device=device)
        # Pass the tolerances to the solver.
        solver_options = {'rtol': rtol, 'atol': atol}
    else:
        raise ValueError(f"Unknown method {method}")

    # Create the ODE function with the current parameter A.
    ode_func = LinearODE(A)

    # Compute the analytic solution.
    y_true = analytic_solution(A.to(torch.float64), x.to(torch.float64), T=T)
    loss_true = y_true.sum()
    loss_true.backward()
    grad_x_true = x.grad.detach().clone()
    grad_A_true = A.grad.detach().clone()
    x.grad.zero_()
    A.grad.zero_()

    with autocast(device_type='cuda',dtype=dtype_f):
        # test call
        # y = ode_func(0.0, x, verbose=True)
        y_num = ode(ode_func, x, t, method=method, **solver_options)
        y_num_final = y_num[-1]  # the numerical solution at t = T
        
        # --- Now compute sensitivities.
        # We choose a simple scalar loss: the sum of the components of y(T)
        loss_num = y_num_final.sum()
        
        # First, compute gradients using the numerical solution.
        # (We must clear any previous gradients.)
        if x.grad is not None:
            x.grad.zero_()
        if ode_func.A.grad is not None:
            ode_func.A.grad.zero_()
        loss_num.backward()
        grad_x_num = x.grad.detach().clone()
        grad_A_num = ode_func.A.grad.detach().clone()
        # print(grad_A_num)

    
    # Compute relative errors in the gradients (sensitivities).
    rel_error_solution = (y_num_final - y_true).norm() / y_true.norm()
    rel_error_grad_x = (grad_x_num - grad_x_true).norm() / grad_x_true.norm()
    rel_error_grad_A = (grad_A_num - grad_A_true).norm() / grad_A_true.norm()

    return rel_error_solution.item(), rel_error_grad_x.item(), rel_error_grad_A.item()


# --- Main experiment ---
if __name__ == '__main__':
    # We will test both single precision and double precision.
    dtypes_f = [ torch.bfloat16, torch.float16]
    dtypes_y = [ torch.bfloat16, torch.float16, torch.float32]

    # We will use three different methods.
    methods = ['euler','rk4']
    packages = ['torchdiffeq','torchmpnode']

    # For fixed-step methods, experiment with various numbers of steps.
    steps_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # For dopri5, experiment with various tolerance levels.
    # steps_list = steps_list[0:1]
    tol_list = [1e-3, 1e-4, 1e-5, 1e-6]

    print("Starting experiments...\n")
    results = []
    for dtype_y in dtypes_y:
        for dtype_f in dtypes_f:
            for method in methods:
                for package in packages:
                    print(f"--- dtype_f: {dtype_f},dtype_y: {dtype_y}  method: {method}, package: {package} ---")
                    ode = odeint if package == 'torchdiffeq' else mpodeint
                    if method in ['euler', 'rk4']:
                        # print(f"--- Method: {method} (fixed-step) ---")
                        for n_steps in steps_list:
                            sol_err, grad_x_err, grad_A_err = run_experiment(ode,method, dtype_f,dtype_y, n_steps=n_steps)
                            print(f"n_steps = {n_steps:3d}  |  "
                                f"sol. relerr: {sol_err:8.2e}  |  "
                                f"∂y/∂x relerr: {grad_x_err:8.2e}  |  "
                                f"∂y/∂A relerr: {grad_A_err:8.2e}")
                            results.append({
                                "dtype_f": str(dtype_f),
                                "dtype_y": str(dtype_y),
                                "method": method,
                                "package": package,
                                "n_steps": n_steps,
                                "sol_relerr": sol_err,
                                "grad_x_relerr": grad_x_err,
                                "grad_A_relerr": grad_A_err,
                            })
                        print()
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("modeleq_demo.csv", index=False)
    print("\nResults saved to modeleq_demo.csv")

                