"""

This script evaluates the numerical accuracy and gradient fidelity of two ODE solvers—standard torchdiffeq's `odeint` 
and the torchmpnote variant `mpodeint` on the  linear ODE y' = A y, where A = 0.5·I. 
We compare three floating‑point precisions (float16, bfloat16, float32) and two step counts (8 and 512) across two integrators (Euler and RK4).  

For each configuration, we first compute a high‑precision reference solution via the matrix exponential at t = 2.0. 
We then solve the ODE under autocast at lower precision, measuring solution and gradient errors against the baseline. 
Additionally, we perform a Taylor decay test by perturbing the initial state directionally and observing how the loss 
error decays with perturbation size, extracting empirical slopes to confirm first‑ and second‑order behavior.  

Generated output files:
- `out/relative_errors.csv`: tabulates relative solution and gradient errors for all methods.
- `out/taylor_slopes.csv`: records empirical error‑decay slopes.
- `out/decay_<method>_<solver>_<precision>_<steps>.csv`: raw (h, error0, error1) data for each decay plot.
- `out/decay_<method>_<solver>_<precision>_<steps>.png`: corresponding log‑log error curves.
- `out/decay_<method>_<solver>_<precision>_<steps>.tex`: TikZ figures (if `tikzplotlib` is installed).

See the `out` directory for detailed results and plots.
"""
import torch
import torch.nn as nn
from torch.amp import autocast
from torchdiffeq import odeint
import os
import sys
import csv
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint

# Configuration
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0, t1     = 0.0, 2.0
dim         = 2
steps_list  = [8,512]
method_list = ['euler', 'rk4']
precisions  = [torch.float16, torch.bfloat16, torch.float32]


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Define a simple linear ODE: dz/dt = A z, with A diagonal matrix
        A = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float32)
        self.theta = nn.Parameter(A)
    def forward(self, t, z):
        # Compute the derivative at time t for state z
        return torch.matmul(self.theta, z)

# prepare data structures to store results and errors
results = []
error_data = {}
for meth in method_list:
    # Initialize error data structure for each method
    error_data[meth] = {
        'steps': [],
        'odeint': {'sol': [], 'grad_z0': [], 'grad_theta': []},
        'mpodeint': {'sol': [], 'grad_z0': [], 'grad_theta': []},
        'taylor': {}
    }
    for p in precisions:
        # For each precision, store Taylor expansion related errors and slopes
        error_data[meth]['taylor'][str(p)] = {
            'slopes_vs_steps': {
                'odeint': {'steps': [], 'slope0': [], 'slope1': []},
                'mpodeint': {'steps': [], 'slope0': [], 'slope1': []}
            },
            'decay_vs_h': {
                'odeint': {},
                'mpodeint': {}
            }
        }

for meth in method_list:
    for N in steps_list:
        # Define time span with N steps
        t_span = torch.linspace(t0, t1, N, device=device, dtype=torch.float32)
        
        # Baseline solution and gradients with full precision
        func_base       = ODEFunc(dim).to(device)
        z0_base         = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
        target          = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)
        # Compute exact solution using matrix exponential
        sol_baseline    = torch.matmul(torch.matrix_exp(func_base.theta * t_span[-1]), z0_base)
        loss_baseline   = 0.5 * ((sol_baseline - target)**2).sum()
        loss_baseline.backward()
        baseline_sol     = sol_baseline.detach().clone()
        baseline_grad_z0 = z0_base.grad.detach().clone()
        baseline_grad_theta = func_base.theta.grad.detach().clone()
        
        for solver_name, ode_fn, store in [('odeint', odeint, 'odeint'),
                                           ('mpodeint', mpodeint, 'mpodeint')]:
            # Run solver under autocast with lowest precision for speed
            with autocast(device_type='cuda', dtype=precisions[0]):
                func_lp     = ODEFunc(dim).to(device)
                z0_lp       = torch.ones((dim,1), dtype=torch.float32, device=device, requires_grad=True)
                sol_lp_list = ode_fn(func_lp, z0_lp, t_span, method=meth)
                sol_lp      = sol_lp_list[-1]
                L_lp        = 0.5 * ((sol_lp - target)**2).sum()
                L_lp.backward()
            
        def rel_err(x,y): 
            # Compute relative error norm between x and y
            return torch.norm(x-y)/torch.norm(y)
            
        # Store relative errors for solutions and gradients
        error_data[meth]['steps'].append(N)
        error_data[meth][store]['sol'].append(rel_err(sol_lp.detach(), baseline_sol).item())
        error_data[meth][store]['grad_z0'].append(rel_err(z0_lp.grad.detach(), baseline_grad_z0).item())
        error_data[meth][store]['grad_theta'].append(rel_err(func_lp.theta.grad.detach(), baseline_grad_theta).item())
        # Taylor decay + slopes computation for different precisions
        for dtype in precisions:
            prec_str = str(dtype)
            for solver_name, ode_fn in [('odeint', odeint), ('mpodeint', mpodeint)]:
                # Define loss function with given solver and precision under autocast
                def loss_fn(x):
                    with autocast(device_type='cuda', dtype=dtype):
                        sol_list = ode_fn(ODEFunc(dim).to(device), x, t_span, method=meth)
                    return 0.5 * ((sol_list[-1] - target)**2).sum()
                
                # Seed RNG for reproducibility
                torch.manual_seed(0)
                x0   = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
                v    = torch.randn_like(x0)
                v   /= torch.norm(v)
                # Compute baseline loss and directional derivative Jv
                with autocast(device_type='cuda', dtype=dtype):
                    L0 = loss_fn(x0)
                    x0.grad = None
                    L0.backward()
                    grad0 = x0.grad.to(torch.float32)
                    Jv    = (grad0.view(-1) @ v.view(-1).to(torch.float32)).item()
                    x0.grad.zero_()
                
                h_vals       = [0.92**i for i in range(50)]
                taylor_err0  = []
                taylor_err1  = []
                # Evaluate Taylor error decay for perturbations h*v
                for h in h_vals:
                    with autocast(device_type='cuda', dtype=dtype):
                        Lp = loss_fn(x0 + h * v)
                    taylor_err0.append(abs((Lp - L0).item()))
                    taylor_err1.append(abs((Lp - L0 - h * Jv).item()))
                
                # Store raw decay data
                error_data[meth]['taylor'][prec_str]['decay_vs_h'][solver_name][N] = (
                    h_vals, taylor_err0, taylor_err1
                )
                # Compute slopes of log-log plots for error decay
                lh   = torch.log(torch.tensor(h_vals))
                le0  = torch.log(torch.tensor(taylor_err0) + 1e-12)
                le1  = torch.log(torch.tensor(taylor_err1) + 1e-12)
                slope0 = ((lh-lh.mean())*(le0-le0.mean())).sum()/((lh-lh.mean())**2).sum()
                slope1 = ((lh-lh.mean())*(le1-le1.mean())).sum()/((lh-lh.mean())**2).sum()
                sd = error_data[meth]['taylor'][prec_str]['slopes_vs_steps'][solver_name]
                sd['steps'].append(N)
                sd['slope0'].append(slope0.item())
                sd['slope1'].append(slope1.item())

# write relative error results to CSV
os.makedirs('out', exist_ok=True)
with open('out/relative_errors.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['method','n_steps','sol_odeint','gz0_odeint','gth_odeint',
                'sol_mp','gz0_mp','gth_mp'])
    
    for meth in method_list:
        for i, N in enumerate(error_data[meth]['steps']):
            r = error_data[meth]
            
            print(f"Method: {meth}, Step Index: {i}, N: {N}")
            print(f"Lengths - sol: {len(r['odeint']['sol'])}, grad_z0: {len(r['odeint']['grad_z0'])}, grad_theta: {len(r['odeint']['grad_theta'])}")
            
            if i >= len(r['odeint']['sol']) or i >= len(r['odeint']['grad_z0']) or i >= len(r['odeint']['grad_theta']):
                print(f"Index {i} out of range for method {meth}. Skipping...")
                continue
            
            w.writerow([meth, N,
                        r['odeint']['sol'][i],  r['odeint']['grad_z0'][i],  r['odeint']['grad_theta'][i],
                        r['mpodeint']['sol'][i], r['mpodeint']['grad_z0'][i], r['mpodeint']['grad_theta'][i]])

# write Taylor slopes to CSV
with open('out/taylor_slopes.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['method','precision','steps','solver','slope0','slope1'])
    for meth in method_list:
        for prec_str, block in error_data[meth]['taylor'].items():
            for solver in ['odeint','mpodeint']:
                d = block['slopes_vs_steps'][solver]
                for i,N in enumerate(d['steps']):
                    w.writerow([meth,prec_str,N,solver,d['slope0'][i],d['slope1'][i]])

# Generate plots of Taylor error decay and save data to CSV and TikZ
for meth in method_list:
    for prec in precisions:
        decay = error_data[meth]['taylor'][str(prec)]['decay_vs_h']
        precision_name = str(prec)
        for solver in ['odeint','mpodeint']:
            for N, (h_vals, e0, e1) in decay[solver].items():
                plt.figure()
                plt.loglog(h_vals, e0, '-', label='E0')
                plt.loglog(h_vals, e1, '--', label='E1')
                plt.xlabel('Perturbation size (r)', fontsize=14)
                plt.ylabel('Error', fontsize=14)
                # plt.title(f"{meth} | {solver} | steps={N} | prec={precision_name}")
                plt.legend(fontsize=14)
                plt.grid(True, which='both', linestyle='--', alpha=0.4)
                plt.tight_layout()
                # Save plot as PNG
                plt.savefig(f"out/decay_{meth}_{solver}_{precision_name}_{N}.png")
                plt.close()
                # Save raw data to CSV for further analysis
                csv_filename = f"out/decay_{meth}_{solver}_{precision_name}_{N}.csv"
                with open(csv_filename, 'w', newline='') as cf:
                    cw = csv.writer(cf)
                    cw.writerow(['h', 'error0', 'error1'])
                    for hv, err0, err1 in zip(h_vals, e0, e1):
                        cw.writerow([hv, err0, err1])
                print(f"Saved decay data CSV: {csv_filename}")
                # Attempt to save plot as TikZ file for LaTeX integration
                try:
                    import tikzplotlib
                    tikzplotlib.save(f"out/decay_{meth}_{solver}_{precision_name}_{N}.tex")
                except ImportError:
                    print("tikzplotlib not installed; skipping TikZ export.")
