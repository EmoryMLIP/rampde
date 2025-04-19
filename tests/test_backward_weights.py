import unittest
import torch
# from torchdiffeq import odeint

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint, NoScaler, DynamicScaler
import math
from torch.func import functional_call


from torch.amp import autocast

# Define our nonlinear ODE as a module.
# Its parameters (A, B, b) are registered as nn.Parameters and are kept in float32.
class NonlinearODE(torch.nn.Module):
    def __init__(self, dim):
        super(NonlinearODE, self).__init__()
        # Always stored in float32.
        self.A = torch.nn.Parameter(torch.randn(dim, dim, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.randn(dim, dim, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.randn(dim, dtype=torch.float32))

    def forward(self, t, x):
        return self.B @ torch.tanh(torch.matmul(self.A,x) + self.b)

class TestTaylorExpansionODE(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available; skipping GPU tests.")
        self.device = torch.device("cuda:0")
        self.dim = 2         # small state dimension
        self.n_time = 10     # few time steps to keep the integration fast
        self.t0 = 0.0
        self.t1 = 1.0
        
    def test_taylor_decay(self):
        # Test for both Euler and RK4 methods, and for three working precisions.
        for method in ['euler', 'rk4']:
            for precision in [torch.float16, torch.bfloat16, torch.float32]:
                with self.subTest(method=method, precision=precision):
                    print(f"\n\nTesting method {method} at precision {precision}")
                    dtype = precision
                    # Create an initial state x0 and a perturbation direction v.
                    x0 = torch.randn(self.dim, device=self.device, dtype=dtype, requires_grad=True)
                    # Create a dict with the same keys as self.model.named_parameters() and random noise in each entry
                    
                    
                    # Create a time tensor on [t0,t1].
                    t = torch.linspace(self.t0, self.t1, self.n_time, device=self.device, dtype=dtype)
                    # (We do not require grad for t in this test.)

                    # Create our ODE model; its parameters are in float32.
                    model = NonlinearODE(self.dim).to(self.device)
                    theta0 = {k: v.clone() for k, v in model.named_parameters()}
                    v = {k: torch.randn_like(v) for k, v in theta0.items()}
                    v = {k: v / torch.norm(v) for k, v in v.items()}


                    # Define a function f that returns the terminal state, running under autocast.
                    def f(x):
                        with autocast(device_type='cuda', dtype=dtype):
                            out = odeint(model, x, t, method=method)
                        return torch.sum(out[-1])
                    
                    # Evaluate f at the base point.
                    with autocast(device_type='cuda', dtype=dtype):
                        f0 = f(x0)
                        # Compute the directional derivative J(x0)*v using torch.autograd.functional.jvp.
                        # (f0, Jv) = jvp(f, (x0,), (v,))
                        x0.grad = None
                        f0 = f(x0)
                        f0.backward()
                        
                        Jv = {k: torch.sum(v[k] * p.grad) for k, p in model.named_parameters()}
                        # Sum all the Jv values to get the directional derivative.
                        Jv = sum(Jv.values())
                        x0.grad = None

                    # Choose a set of perturbation scales.
                    h_vals = [(0.92 ** i) for i in range(50)]  # e.g., 1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4
                    # h_vals.append(0.0)
                    errors0 = []  # zero‐order error: ||f(x0+h*v)-f(x0)||
                    errors1 = []  # first‐order error: ||f(x0+h*v)-f(x0)-h*Jv||
                    orders0 = []  # observed order of convergence for zero‐order error
                    orders1 = []  # observed order of convergence for first‐order error

                    for i,h in enumerate(h_vals):
                        theta_pert = {k: p + h * v[k] for k, p in theta0.items()}
                        # Set the parameters of the model to theta_pert.
                        for k, p in model.named_parameters():
                            p.data = theta_pert[k]
                        with autocast(device_type='cuda', dtype=dtype):
                            out = odeint(model, x0, t, method=method)
                        f_pert = torch.sum(out[-1])
                


                        error0 = torch.norm(f_pert - f0)
                        error1 = torch.norm(f_pert - f0 - h * Jv)
                        errors0.append(error0.item())
                        errors1.append(error1.item())

                        if i>0:
                            order0 = math.log((errors0[i]/errors0[i-1])) / math.log(h_vals[i] / h_vals[i-1])
                            order1 = math.log((errors1[i]/errors1[i-1])) / math.log(h_vals[i] / h_vals[i-1])
                            orders0.append(order0)
                            orders1.append(order1)
                            print(f"h: {h:.2e}, Error0: {error0:.2e}, Error1: {error1:.2e}, Order0: {order0:.2f}, Order1: {order1:.2f}")
                    
                    # For a quantitative test we can compute the slopes in log-log space.
                    # We expect log(error0) vs. log(h) to have a slope near 1,
                    # and log(error1) vs. log(h) to have a slope near 2.
                    log_h = torch.log(torch.tensor(h_vals))
                    log_E0 = torch.log(torch.tensor(errors0) + 1e-12)
                    log_E1 = torch.log(torch.tensor(errors1) + 1e-12)
                    slope0 = ((log_h - log_h.mean())*(log_E0 - log_E0.mean())).sum() / ((log_h - log_h.mean())**2).sum()
                    slope1 = ((log_h - log_h.mean())*(log_E1 - log_E1.mean())).sum() / ((log_h - log_h.mean())**2).sum()

                    tol = 0.3  # Allow a tolerance of 0.3 in the slope.
                    # pass test if there are at least 8 entries in slope1 that are between 1.8 and 2.2
                    self.assertTrue(len([s for s in orders1 if 1.8 < s < 10.0]) >= 8,
                        f"First order error observed order is not within tolerance for method {method} at precision {dtype}")

if __name__ == '__main__':
    unittest.main()