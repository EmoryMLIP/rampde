"""
This script benchmarks gradient fidelity in low‑precision time‑integration.
We integrate the linear ODE ẏ = Ay with A = 0.5 I₂ and compare three
implementations:

1. **torchdiffeq** (`odeint`) executed inside a CUDA autocast region
   (`float16`, `bfloat16`, or `float32`).
2. **torchmpnode** (`mpodeint`) which performs the same Runge–Kutta steps
   but keeps internal high‑precision accumulators.
3. A hand‑rolled “pseudo‑autocast” Euler solver that mimics mixed precision
   by down‑casting every stage matrix–vector product.

For each integrator we measure the relative error of
(a) the final state  y(T),
(b) ∇_{z₀} ℒ,  and
(c) ∇_{θ} ℒ,
where the loss ℒ = ½‖y(T) − 2‖² admits a closed‑form gradient.  The exact
solution computed with a matrix exponential serves as reference.

We sweep over a logarithmic grid of step counts (32 – 8192) and two explicit
methods (forward Euler, RK4).  All results are written to
`relative_errors.csv`; three log–log PNGs visualise the error curves.

Run on GPU with:
    python test_autocast_odeint.py
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from torchdiffeq import odeint
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint
import csv
import matplotlib.pyplot as plt

prec = torch.float16
num_iters = 1    # global number of “training” iterations
lr = 1e-3         # learning rate for updating θ

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        A = torch.tensor([[0.5, 0.0],
                          [0.0, 0.5]], dtype=torch.float32)
        self.theta = nn.Parameter(A)
    
    def forward(self, t, z):
        """Compute the time derivative dz/dt = theta * z."""

        with autocast(device_type='cuda', dtype=prec):
            return torch.matmul(self.theta, z)



def euler_forward_manual_pseudoac(z0, func, t0, t1, N):
    h = (t1 - t0) / (N - 1)
    z_list = []
    current_z = z0.clone() 
    t = t0
    z_list.append(current_z)
    for _ in range(N - 1):
        z_half = current_z
        f_val_half = torch.matmul(func.theta.to(prec), z_half.to(prec))
        h_half = torch.tensor(h, dtype=prec, device=current_z.device)
        z_next_half = z_half + (h_half * f_val_half).float()
        z_next = z_next_half
        z_list.append(z_next)
        current_z = z_next
        t += h
    return z_list

def manual_pseudoac(z_list, func, t0, t1):
    N = len(z_list) - 1
    h = (t1 - t0) / N
    h = torch.tensor(h, dtype=torch.float32, device=z_list[0].device)
    dim = func.theta.shape[0]
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=z_list[0].device)
    
    dL_dz_half = [None]*(N+1)
    dL_dz_half[N] = z_list[-1] - target
    dL_dtheta_half = torch.zeros_like(func.theta)
    
    for k in reversed(range(N)):
        factor_half = ((h * func.theta).transpose(0,1)).to(prec)
        dL_dz_half[k] = dL_dz_half[k+1].float() + torch.matmul(factor_half, dL_dz_half[k+1].to(prec))
        dL_dtheta_half = (dL_dtheta_half.to(prec) 
                          + (h * torch.matmul(dL_dz_half[k+1].float(),
                                              z_list[k].float().transpose(0,1))).to(prec))
        dL_dtheta_half = dL_dtheta_half.float()
    return dL_dz_half[0], dL_dtheta_half

# Full precision analytical solution
def run_baseline(method, N, device, t0, t1, dim):
    """Run baseline full-precision integration and compute gradients."""
    t_span = torch.linspace(t0, t1, N, device=device)
    func   = ODEFunc(dim).to(device)
    z0     = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)

    optim = torch.optim.SGD(func.parameters(), lr=lr)
    for _ in range(num_iters):
        optim.zero_grad()
        if z0.grad is not None:
            z0.grad.zero_()
        sol  = torch.matmul(torch.matrix_exp(func.theta * t_span[-1]), z0)
        loss = 0.5 * ((sol - target)**2).sum()
        loss.backward()
        optim.step()

    return sol.detach().clone(), z0.grad.detach().clone(), func.theta.grad.detach().clone()

# Run torchdiffeq under autocast
def run_autocast_odeint(method, N, device, t0, t1, dim, prec):
    """Run torchdiffeq odeint inside autocast region and compute gradients."""
    t_span = torch.linspace(t0, t1, N, device=device)
    func   = ODEFunc(dim).to(device)
    z0     = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)

    optim = torch.optim.SGD(func.parameters(), lr=lr)
    for _ in range(num_iters):
        optim.zero_grad()
        if z0.grad is not None:
            z0.grad.zero_()
        with autocast(device_type='cuda', dtype=prec):
            sol  = odeint(func, z0, t_span, method=method)
            loss = 0.5 * ((sol[-1] - target)**2).sum()
        loss.backward()
        optim.step()

    return sol[-1].detach().clone(), z0.grad.detach().clone(), func.theta.grad.detach().clone()

# Run torchmpnode under autocast
def run_autocast_mpodeint(method, N, device, t0, t1, dim, prec):
    """Run torchmpnode odeint inside autocast region and compute gradients."""

    t_span = torch.linspace(t0, t1, N, device=device)
    func   = ODEFunc(dim).to(device)
    z0     = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)

    optim = torch.optim.SGD(func.parameters(), lr=lr)
    for _ in range(num_iters):
        optim.zero_grad()
        if z0.grad is not None:
            z0.grad.zero_()
        with autocast(device_type='cuda', dtype=prec):
            sol  = mpodeint(func, z0, t_span, method=method)
            loss = 0.5 * ((sol[-1] - target)**2).sum()
        loss.backward()
        optim.step()

    return sol[-1].detach().clone(), z0.grad.detach().clone(), func.theta.grad.detach().clone()


device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0, t1       = 0.0, 2.0
dim          = 2
steps_list   = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
method_list  = ['rk4']  # 'euler', etc.

results    = []
error_data = {meth: {'steps': [],
                     'odeint': {'sol': [], 'grad_z0': [], 'grad_theta': []},
                     'mpodeint': {'sol': [], 'grad_z0': [], 'grad_theta': []}}
              for meth in method_list}

for meth in method_list:
    print("Processing method:", meth)
    for N in steps_list:
        baseline_sol, baseline_grad_z0, baseline_grad_theta = run_baseline(meth, N, device, t0, t1, dim)
        ac_sol, ac_grad_z0, ac_grad_theta             = run_autocast_odeint(meth, N, device, t0, t1, dim, prec)
        acmp_sol, acmp_grad_z0, acmp_grad_theta       = run_autocast_mpodeint(meth, N, device, t0, t1, dim, prec)

        rel_err_sol_odeint        = torch.norm(ac_sol - baseline_sol)        / torch.norm(baseline_sol)
        rel_err_grad_z0_odeint    = torch.norm(ac_grad_z0 - baseline_grad_z0)    / torch.norm(baseline_grad_z0)
        rel_err_grad_theta_odeint = torch.norm(ac_grad_theta - baseline_grad_theta) / torch.norm(baseline_grad_theta)

        rel_err_sol_mpodeint        = torch.norm(acmp_sol - baseline_sol)        / torch.norm(baseline_sol)
        rel_err_grad_z0_mpodeint    = torch.norm(acmp_grad_z0 - baseline_grad_z0)    / torch.norm(baseline_grad_z0)
        rel_err_grad_theta_mpodeint = torch.norm(acmp_grad_theta - baseline_grad_theta) / torch.norm(baseline_grad_theta)

        results.append([
            meth, N,
            rel_err_sol_odeint.item(),        rel_err_grad_z0_odeint.item(),        rel_err_grad_theta_odeint.item(),
            rel_err_sol_mpodeint.item(),      rel_err_grad_z0_mpodeint.item(),      rel_err_grad_theta_mpodeint.item()
        ])

        error_data[meth]['steps'].append(N)
        error_data[meth]['odeint']['sol'].append(rel_err_sol_odeint.item())
        error_data[meth]['odeint']['grad_z0'].append(rel_err_grad_z0_odeint.item())
        error_data[meth]['odeint']['grad_theta'].append(rel_err_grad_theta_odeint.item())
        error_data[meth]['mpodeint']['sol'].append(rel_err_sol_mpodeint.item())
        error_data[meth]['mpodeint']['grad_z0'].append(rel_err_grad_z0_mpodeint.item())
        error_data[meth]['mpodeint']['grad_theta'].append(rel_err_grad_theta_mpodeint.item())

        print(
            f"Method: {meth}, Steps: {N}\n"
            f"  torchdiffeq: sol error = {rel_err_sol_odeint:.2e}, dL/dz0 error = {rel_err_grad_z0_odeint:.2e}, dL/dtheta error = {rel_err_grad_theta_odeint:.2e}\n"
            f"  torchmpnode: sol error = {rel_err_sol_mpodeint:.2e}, dL/dz0 error = {rel_err_grad_z0_mpodeint:.2e}, dL/dtheta error = {rel_err_grad_theta_mpodeint:.2e}\n"
        )


result_dir = os.path.join(os.path.dirname(__file__), 'results_multistep')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


csv_filename = os.path.join(result_dir, "relative_errors.csv")
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["method", "n_steps", 
                     "rel_err_sol_odeint", "rel_err_grad_z0_odeint", "rel_err_grad_theta_odeint",
                     "rel_err_sol_mpodeint", "rel_err_grad_z0_mpodeint", "rel_err_grad_theta_mpodeint"])
    writer.writerows(results)
print("CSV file saved as", csv_filename)

for meth in method_list:
    steps = error_data[meth]['steps']
    # Plot relative error for final solution (z_T)
    plt.figure()
    plt.plot(steps, error_data[meth]['odeint']['sol'], marker='o', label='torchdiffeq (autocast)', color='blue')
    plt.plot(steps, error_data[meth]['mpodeint']['sol'], marker='s', label='torchmpnode (autocast)', color='red')
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlabel('Number of Steps')
    # plt.ylabel('Relative Error (Final Solution)')
    # plt.title(f"Relative Error in Final Solution - Method: {meth}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(result_dir, f'relative_error_solutionf_{meth}.png'))

    # Plot relative error for gradient wrt initial condition (dL/dz0)
    plt.figure()
    plt.plot(steps, error_data[meth]['odeint']['grad_z0'], marker='o', label='torchdiffeq (autocast)', color='blue')
    plt.plot(steps, error_data[meth]['mpodeint']['grad_z0'], marker='s', label='torchmpnode (autocast)', color='red')
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlabel('Number of Steps')
    # plt.ylabel('Relative Error (dL/dz0)')
    # plt.title(f"Relative Error in dL/dz0 - Method: {meth}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(result_dir, f'relative_error_grad_z0f_{meth}.png'))

    # Plot relative error for gradient wrt theta (dL/dtheta)
    plt.figure()
    plt.plot(steps, error_data[meth]['odeint']['grad_theta'], marker='o', label='torchdiffeq (autocast)', color='blue')
    plt.plot(steps, error_data[meth]['mpodeint']['grad_theta'], marker='s', label='torchmpnode (autocast)', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.xlabel('Number of Steps')
    # plt.ylabel('Relative Error (dL/dtheta)')
    # plt.title(f"Relative Error in dL/dtheta - Method: {meth}")
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'relative_error_grad_thetaf_{meth}.png'))

