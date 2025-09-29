import unittest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rampde import odeint
import math

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
        # The t-dependence is included to test differentiation with respect to time
        return self.B @ torch.tanh(self.A @ x + t**2 * self.b)

class TestTaylorExpansionODE(unittest.TestCase):
    results = []  # class-level list to store test results
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available; skipping GPU tests.")
        self.device = torch.device("cuda:0")
        self.dim = 2         # small state dimension
        self.n_time = 10     # few time steps to keep the integration fast
        self.t0 = 0.0
        self.t1 = 1.0

    def _run_taylor_test(self, method, precision, scale_input=0, scale_weights=0, scale_time=0):
        """
        Runs a Taylor test with selectable differentiation directions.
        
        Args:
            method (str): The ODE integration method ('euler' or 'rk4')
            precision (torch.dtype): The working precision for the test
            scale_input (float): Scale factor for input differentiation (0 or 1)
            scale_weights (float): Scale factor for weights differentiation (0 or 1)
            scale_time (float): Scale factor for time differentiation (0 or 1)
        """
        test_name = []
        if scale_input: test_name.append("input")
        if scale_weights: test_name.append("weights")
        if scale_time: test_name.append("time")
        test_name = "+".join(test_name) if test_name else "none"
        
        quiet = os.environ.get("TORCHMPNODE_TEST_QUIET", "0") == "1"
        if not quiet:
            print(f"\n\nTesting method {method} at precision {precision} for {test_name}")
        dtype = precision
        
        # Create our ODE model; its parameters are in float32.
        model = NonlinearODE(self.dim).to(self.device)
        
        # Create an initial state x0 with requires_grad if we're differentiating wrt input
        x0 = torch.randn(self.dim, device=self.device, dtype=dtype, requires_grad=(scale_input > 0))
        
        # Create a time tensor on [t0,t1] with requires_grad if we're differentiating wrt time
        t = torch.linspace(self.t0, self.t1, self.n_time, device=self.device, dtype=dtype, requires_grad=(scale_time > 0))
        
        # Create perturbation vectors for each variable
        # Input perturbation
        v_x = torch.randn(self.dim, device=self.device, dtype=dtype)
        v_x = v_x / torch.norm(v_x) * scale_input  # Scale by input factor
        
        # Weight perturbation
        theta0 = {k: v.clone() for k, v in model.named_parameters()}
        v_theta = {k: torch.randn_like(v) for k, v in theta0.items()}
        v_theta_norm = torch.sqrt(sum(torch.sum(v_i ** 2) for v_i in v_theta.values()))
        v_theta = {k: v_i / v_theta_norm * scale_weights for k, v_i in v_theta.items()}
        
        # Time perturbation
        v_t = .45 * (torch.rand_like(t) - 0.5) * ((self.t1 - self.t0) / self.n_time)
        v_t = v_t / torch.norm(v_t) * scale_time
        
        # Define a single function to evaluate the ODE
        def f(x, t_input, params=None):
            # Set model parameters if provided
            if params:
                for k, p in model.named_parameters():
                    p.data = params[k]
            
            with autocast(device_type='cuda', dtype=dtype):
                out = odeint(model, x, t_input, method=method)
            return torch.sum(out[-1])
        
        # Base point evaluation
        f0 = f(x0, t)
        
        # set gradients to None before computing directional derivatives
        if scale_input != 0:
            x0.grad = None
        if scale_weights != 0:
            for p in model.parameters():
                p.grad = None
        if scale_time != 0:
            t.grad = None
        f0.backward()

        Jv = 0.0
        if scale_input != 0:
            Jv += torch.sum(x0.grad * v_x)
        if scale_weights != 0:
            Jv += sum(torch.sum(v_theta[k] * p.grad) for k, p in model.named_parameters()) 
        if scale_time != 0:    
            Jv += torch.sum(t.grad * v_t)

        # Run the Taylor test
        h_vals = [(0.95 ** i) for i in range(50)]
        errors0 = []  # zero‐order error: ||f(x0+h*v)-f(x0)||
        errors1 = []  # first‐order error: ||f(x0+h*v)-f(x0)-h*Jv||
        orders0 = []  # observed order of convergence for zero‐order error
        orders1 = []  # observed order of convergence for first‐order error

        for i, h in enumerate(h_vals):
            # Apply perturbations to all variables (scaled by their respective factors)
            x_pert = x0 + h * v_x
            t_pert = t + h * v_t
            theta_pert = {k: p + h * v_theta[k] for k, p in theta0.items()}
            
            # Evaluate perturbed function
            f_pert = f(x_pert, t_pert, theta_pert)
            
            error0 = torch.norm(f_pert - f0)
            error1 = torch.norm(f_pert - f0 - h * Jv)
            errors0.append(error0.item())
            errors1.append(error1.item())

            if i > 0:
                eps = 1e-20
                order0 = math.log((errors0[i]+eps)/(errors0[i-1]+eps)) / math.log(h_vals[i] / h_vals[i-1])
                order1 = math.log((errors1[i]+eps)/(errors1[i-1]+eps)) / math.log(h_vals[i] / h_vals[i-1])
                orders0.append(order0)
                orders1.append(order1)
                if not quiet:
                    print(f"h: {h:.2e}, Error0: {error0:.2e}, Error1: {error1:.2e}, Order0: {order0:.2f}, Order1: {order1:.2f}")
        
        # For a quantitative test we can compute the slopes in log-log space.
        # We expect log(error0) vs. log(h) to have a slope near 1,
        # and log(error1) vs. log(h) to have a slope near 2.
        log_h = torch.log(torch.tensor(h_vals))
        log_E0 = torch.log(torch.tensor(errors0) + 1e-12)
        log_E1 = torch.log(torch.tensor(errors1) + 1e-12)
        slope0 = ((log_h - log_h.mean())*(log_E0 - log_E0.mean())).sum() / ((log_h - log_h.mean())**2).sum()
        slope1 = ((log_h - log_h.mean())*(log_E1 - log_E1.mean())).sum() / ((log_h - log_h.mean())**2).sum()

        # pass test if there are at least 8 entries in slope1 that are between 1.8 and arbitrary high value (to avoid instability)
        passed = len([s for s in orders1 if 1.8 < s < 10.0]) >= 7
        # Store result for reporting
        self.__class__.results.append({
            'method': method,
            'precision': str(precision),
            'input': int(scale_input != 0),
            'weights': int(scale_weights != 0),
            'time': int(scale_time != 0),
            'pass': passed
        })
        self.assertTrue(passed,
            f"First order error observed order is not within tolerance for method {method} at precision {dtype} for {test_name} test:  { len([s for s in orders1 if 1.8 < s < 10.0]) } < 7")
    

def _add_test(method, precision, scale_input, scale_weights, scale_time):
    def test(self):
        self._run_taylor_test(method, precision, scale_input, scale_weights, scale_time)
    name = f"test_{method}_{str(precision).replace('torch.','')}_input{int(scale_input)}_weights{int(scale_weights)}_time{int(scale_time)}"
    setattr(TestTaylorExpansionODE, name, test)

# Add all test cases (no for-loops in test methods)
for method in ['euler', 'rk4']:
    for precision in [torch.float16, torch.float32]:
        _add_test(method, precision, 1, 0, 0)  # input
        _add_test(method, precision, 0, 1, 0)  # weights
        _add_test(method, precision, 0, 0, 1)  # time
        # Combined random direction
        import random
        import numpy as np
        np.random.seed(42)
        scales = np.random.rand(3)
        scales = scales / np.linalg.norm(scales)
        _add_test(method, precision, float(scales[0]), float(scales[1]), float(scales[2]))

@classmethod
def tearDownClass(cls):
    # Print a report table at the end
    print("\n\nTaylor Test Report:")
    print(f"{'method':<8} {'precision':<10} {'input':<5} {'weights':<7} {'time':<5} {'pass':<5}")
    for r in cls.results:
        print(f"{r['method']:<8} {r['precision']:<10} {r['input']:<5} {r['weights']:<7} {r['time']:<5} {str(r['pass']):<5}")
TestTaylorExpansionODE.tearDownClass = tearDownClass


if __name__ == '__main__':
    unittest.main()
