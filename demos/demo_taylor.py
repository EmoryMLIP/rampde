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

# --- configuration in the Taylor expansion of the ODE function---
# Choose between 'input' and 'weights'
# 'input' means the perturbation is applied to the input of the ODE function
# 'weights' means the perturbation is applied to the weights of the ODE function
config = 'input'  # 'input' or 'weights'

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0, t1 = 0.0, 2.0
dim = 2
steps_list = [8, 512]
method_list = ['euler', 'rk4']
precisions = ['float16', 'float32', 'tfloat32'] #'bfloat16', 


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Define a simple linear ODE: dz/dt = A z, with A diagonal matrix
        A = torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float32)
        self.theta = nn.Parameter(A)
    def forward(self, t, z):
        return torch.matmul(self.theta, z)

# Data structure
error_data = {meth: {
    'steps': [],
    'odeint': {'sol': [], 'grad_z0': [], 'grad_theta': []},
    'mpodeint': {'sol': [], 'grad_z0': [], 'grad_theta': []},
    'taylor': {}
} for meth in method_list}

# Main loop
for meth in method_list:
    for p in precisions:
        # For each precision, store Taylor expansion related errors and slopes
        error_data[meth]['taylor'][str(p)] = {
            'slopes_vs_steps': {
                'odeint': {'steps': [], 'slope0': [], 'slope1': []},
                'mpodeint': {'steps': [], 'slope0': [], 'slope1': []}
            },
            'decay_vs_h': {'odeint': {}, 'mpodeint': {}}
        }

    for N in steps_list:
        # Define time span with N steps
        t_span = torch.linspace(t0, t1, N, device=device, dtype=torch.float32)

        # Baseline solution and gradients with full precision
        func_base = ODEFunc(dim).to(device)
        z0_base = torch.ones((dim,1), dtype=torch.float32, device=device, requires_grad=True)
        target = torch.full((dim,1), 2.0, dtype=torch.float32, device=device)

        # Compute exact solution using matrix exponential
        sol_baseline = torch.matmul(torch.matrix_exp(func_base.theta * t_span[-1]), z0_base)
        loss_base = 0.5 * ((sol_baseline - target)**2).sum()
        loss_base.backward()
        baseline_sol = sol_baseline.detach().clone()
        baseline_grad_z0 = z0_base.grad.detach().clone()
        baseline_grad_theta = func_base.theta.grad.detach().clone()
        error_data[meth]['steps'].append(N)

        for solver_name, ode_fn, store in [('odeint', odeint, 'odeint'), ('mpodeint', mpodeint, 'mpodeint')]:
            with autocast(device_type='cuda', dtype=torch.float16):
                func_lp = ODEFunc(dim).to(device)
                z0_lp = torch.ones((dim,1), dtype=torch.float32, device=device, requires_grad=True)
                sol_lp = ode_fn(func_lp, z0_lp, t_span, method=meth)[-1]
                L_lp = 0.5 * ((sol_lp - target)**2).sum()
                L_lp.backward()

            def rel_err(x, y): return torch.norm(x - y) / torch.norm(y)

            error_data[meth][store]['sol'].append(rel_err(sol_lp.detach(), baseline_sol).item())
            error_data[meth][store]['grad_z0'].append(rel_err(z0_lp.grad.detach(), baseline_grad_z0).item())
            error_data[meth][store]['grad_theta'].append(rel_err(func_lp.theta.grad.detach(), baseline_grad_theta).item())



        # Taylor decay + slopes computation for different precisions
        for prec_str in precisions:
            if prec_str == 'tfloat32':
                dtype = torch.float32
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Using TF32 (tfloat32) backend mode")
            elif prec_str == 'float32':
                dtype = torch.float32
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                print("Using strict float32")
            else:
                dtype = getattr(torch, prec_str)

            # Store errors using string key
            error_data[meth]['taylor'][prec_str] = {
                'slopes_vs_steps': {
                    'odeint': {'steps': [], 'slope0': [], 'slope1': []},
                    'mpodeint': {'steps': [], 'slope0': [], 'slope1': []}
                },
                'decay_vs_h': {'odeint': {}, 'mpodeint': {}}
            }

            for solver_name, ode_fn in [('odeint', odeint), ('mpodeint', mpodeint)]:
                if config == 'input':
                    x0 = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn_like(x0); v /= torch.norm(v)

                    def loss_fn(x):
                        with autocast(device_type='cuda', dtype=dtype):
                            sol = ode_fn(ODEFunc(dim).to(device), x, t_span, method=meth)[-1]
                            return 0.5 * ((sol - target)**2).sum()

                    with autocast(device_type='cuda', dtype=dtype):
                        L0 = loss_fn(x0)
                    x0.grad = None
                    L0.backward()
                    grad0 = x0.grad.to(torch.float32)
                    Jv = (grad0.view(-1) @ v.view(-1).to(torch.float32)).item()
                    x0.grad.zero_()

                    def perturb_fn(h): return loss_fn(x0 + h * v)

                elif config == 'weights':
                    func = ODEFunc(dim).to(device)
                    theta0 = func.theta.detach().clone()
                    torch.manual_seed(0)
                    v = torch.randn_like(theta0); v /= torch.norm(v)

                    with autocast(device_type='cuda', dtype=dtype):
                        sol = ode_fn(func, z0_base, t_span, method=meth)[-1]
                        L0 = 0.5 * ((sol - target)**2).sum()
                    L0.backward()
                    grad0 = func.theta.grad.to(torch.float32)
                    Jv = (grad0.view(-1) @ v.view(-1)).item()

                    def perturb_fn(h):
                        func_h = ODEFunc(dim).to(device)
                        with torch.no_grad():
                            func_h.theta.copy_(theta0 + h * v)
                        with autocast(device_type='cuda', dtype=dtype):
                            sol_h = ode_fn(func_h, z0_base, t_span, method=meth)[-1]
                            return 0.5 * ((sol_h - target)**2).sum()

                h_vals = [0.92**i for i in range(50)]
                err0_list, err1_list = [], []

                # Evaluate Taylor error decay for perturbations h*v
                for h in h_vals:
                    Lh = perturb_fn(h)
                    err0_list.append(abs((Lh - L0).item()))
                    err1_list.append(abs((Lh - L0 - h * Jv).item()))

                error_data[meth]['taylor'][prec_str]['decay_vs_h'][solver_name][N] = (
                    h_vals, err0_list, err1_list
                )

                # Compute slopes of log-log plots for error decay
                lh = torch.log(torch.tensor(h_vals))
                le0 = torch.log(torch.tensor(err0_list) + 1e-12)
                le1 = torch.log(torch.tensor(err1_list) + 1e-12)
                slope0 = ((lh-lh.mean())*(le0-le0.mean())).sum()/((lh-lh.mean())**2).sum()
                slope1 = ((lh-lh.mean())*(le1-le1.mean())).sum()/((lh-lh.mean())**2).sum()

                sd = error_data[meth]['taylor'][prec_str]['slopes_vs_steps'][solver_name]
                sd['steps'].append(N)
                sd['slope0'].append(slope0.item())
                sd['slope1'].append(slope1.item())

# --- write results ---
outdir = 'out_' + config
os.makedirs(outdir, exist_ok=True)

# relative errors
with open(f'{outdir}/relative_errors.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['method','n_steps','sol_odeint','gz0_odeint','gth_odeint','sol_mp','gz0_mp','gth_mp'])
    for meth in method_list:
        for i, N in enumerate(error_data[meth]['steps']):
            w.writerow([
                meth, N,
                error_data[meth]['odeint']['sol'][i],
                error_data[meth]['odeint']['grad_z0'][i],
                error_data[meth]['odeint']['grad_theta'][i],
                error_data[meth]['mpodeint']['sol'][i],
                error_data[meth]['mpodeint']['grad_z0'][i],
                error_data[meth]['mpodeint']['grad_theta'][i]
            ])

# Taylor slopes
with open(f'{outdir}/taylor_slopes.csv','w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['method','precision','steps','solver','slope0','slope1'])
    for meth in method_list:
        for prec_str, block in error_data[meth]['taylor'].items():
            for solver in ['odeint','mpodeint']:
                d = block['slopes_vs_steps'][solver]
                for i, N in enumerate(d['steps']):
                    w.writerow([meth, prec_str, N, solver, d['slope0'][i], d['slope1'][i]])

# Generate plots of Taylor error decay and save data to CSV and TikZ
for meth in method_list:
    for prec in precisions:
        precision_name = str(prec)
        decay = error_data[meth]['taylor'][precision_name]['decay_vs_h']
        prefix = "decayA" if config == 'weights' else "decay"
        for solver in ['odeint','mpodeint']:
            for N, (h_vals, e0, e1) in decay[solver].items():
                plt.figure()
                plt.loglog(h_vals, e0, '-', label='E0')
                plt.loglog(h_vals, e1, '--', label='E1')
                plt.xlabel('Perturbation size (h)')
                plt.ylabel('Error')
                plt.legend()
                plt.grid(True, which='both', linestyle='--', alpha=0.4)
                plt.tight_layout()
                plt.savefig(f"{outdir}/{prefix}_{meth}_{solver}_{precision_name}_{N}.png")
                plt.close()

                with open(f"{outdir}/{prefix}_{meth}_{solver}_{precision_name}_{N}.csv", 'w', newline='') as cf:
                    cw = csv.writer(cf)
                    cw.writerow(['h', 'error0', 'error1'])
                    for hv, err0, err1 in zip(h_vals, e0, e1):
                        cw.writerow([hv, err0, err1])

                try:
                    import tikzplotlib
                    tikzplotlib.save(f"{outdir}/{prefix}_{meth}_{solver}_{precision_name}_{N}.tex")
                except ImportError:
                    print("tikzplotlib not installed; skipping TikZ export.")
