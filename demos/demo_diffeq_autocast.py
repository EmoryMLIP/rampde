# ------------------------------------------------------------
# This script compares torchdiffeq.odeint under autocast 
# and manual “pseudo‐autocast” Euler integrator & backward. 
# ------------------------------------------------------------


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

class ODEFunc(nn.Module):
    """
    A simple ODE dz/dt = A z,
    where A is a learnable 2×2 matrix (stored in float32).
    """
    def __init__(self, dim):
        super().__init__()
        A = torch.tensor([[-0.5, 0.0],
                                     [0.0, -1.5]], dtype=torch.float32)

        self.theta = nn.Parameter(A)
    
    def forward(self, t, z):
        return torch.matmul(self.theta, z)


def euler_forward_manual_pseudoac(z0, func, t0, t1, N):
    """
    Forward Euler in “pseudo‐autocast” mode:
    - matmul and step‐size cast into low precision
    - step computed in low precision then converted back to float32
    """
        
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

    """
    Manual backward pass that mimics autocast+torchdiffeq:
      - Uses low precsion for the matmul‐update accumulation
      - Casts FP32 updates once to low precsion
      - Performs all additions in low precsion
      - Casts final dθ back to FP32.
    Returns:
      dL/dz0 (float32 Tensor), dL/dθ (float32 Tensor)
    """

    N = len(z_list) - 1
    h = (t1 - t0) / N
    h = torch.tensor(h, dtype=torch.float32, device=z_list[0].device)
    dim = func.theta.shape[0]
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=z_list[0].device)
    I = torch.eye(dim, dtype=torch.float32, device=z_list[0].device)
    
    dL_dz_half = [None]*(N+1)
    dL_dz_half[N] = z_list[-1] - target
    dL_dtheta_half = torch.zeros_like(func.theta)
    
    for k in reversed(range(N)):
        factor_half = ((h * func.theta).transpose(0,1)).to(prec)
        dL_dz_half[k] = dL_dz_half[k+1].float() + torch.matmul(factor_half, dL_dz_half[k+1].to(prec))
        # if k == N-1:
        # dL_dtheta_half = (dL_dtheta_half 
        #                 + (h * torch.matmul(dL_dz_half[k+1].float(),
        #                                     z_list[k].float().transpose(0,1))).to(prec))
        # else:

        dL_dtheta_half = (dL_dtheta_half.to(prec) 
                        + (h * torch.matmul(dL_dz_half[k+1].float(),
                                            z_list[k].float().transpose(0,1))).to(prec))
        dL_dtheta_half = dL_dtheta_half.float()
    return dL_dz_half[0], dL_dtheta_half#, dL_dz_half[N]

# Full precision
def run_baseline(method, N, device, t0, t1, dim):
    t_span = torch.linspace(t0, t1, N, device=device)
    func = ODEFunc(dim).to(device)

    z0 = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)
    # sol = odeint(func, z0, t_span, method=method)[-1]
    # Analytical solution for the ODE: z(t) = exp(theta * t) * z0
    sol = torch.matmul(torch.matrix_exp(func.theta * t_span[-1]), z0)
    loss = 0.5 * ((sol - target)**2).sum()
    loss.backward()
    # Return the final solution and the gradients.
    return sol.detach().clone(), z0.grad.detach().clone(), func.theta.grad.detach().clone()

# Run torchdiffeq under autocast
def run_autocast_odeint(method, N, device, t0, t1, dim, prec):
    """
    Mixed‐precision via torchdiffeq’s odeint under autocast:
      - All ops eligible for autocast run in low precision
      - torchdiffeq handles its own backward in mixed precision
    """
    t_span = torch.linspace(t0, t1, N, device=device)
    func = ODEFunc(dim).to(device)
    z0 = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)
    with autocast(device_type='cuda', dtype=prec):
        sol = odeint(func, z0, t_span, method=method)
        loss = 0.5 * ((sol[-1] - target)**2).sum()
        loss.backward()

    return sol[-1].detach().clone(), z0.grad.detach().clone(), func.theta.grad.detach().clone()#, zt.grad.detach().clone()

# Run torchmpnode under autocast
def run_autocast_mpodeint(method, N, device, t0, t1, dim, prec):
    """
    Mixed‐precision via torchmpnode’s odeint under autocast
    """
    t_span = torch.linspace(t0, t1, N, device=device)
    func = ODEFunc(dim).to(device)
    z0 = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)
    with autocast(device_type='cuda', dtype=prec):
        sol = mpodeint(func, z0, t_span, method=method)
        loss = 0.5 * ((sol[-1] - target)**2).sum()
        loss.backward()
    return sol[-1].detach().clone(), z0.grad.detach().clone(), func.theta.grad.detach().clone()

def pseudo_autocast(N, device, t0, t1, dim):
    """
    Manual forward and backward that mirrors autocast behavior
    """
    z0_manual_noac = torch.ones((dim, 1), dtype=torch.float32, device=device)
    func_manual_noac = ODEFunc(dim).to(device)
    z_list_manual_noac = euler_forward_manual_pseudoac(z0_manual_noac, func_manual_noac, t0, t1, N)
    dL_dz0_manual_noac, dL_dtheta_manual_noac = manual_pseudoac(z_list_manual_noac, func_manual_noac, t0, t1)
    
    return z_list_manual_noac[-1].detach().clone(), dL_dz0_manual_noac, dL_dtheta_manual_noac



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0, t1 = 0.0, 2.0
dim = 2
steps_list = [32, 64, 128, 256, 512, 1024]
method_list = ['euler']

results = []  
error_data = {}
for meth in method_list:
    error_data[meth] = {
        'steps': [],
        'odeint': {'sol': [], 'grad_z0': [], 'grad_theta': []},
        # 'mpodeint': {'sol': [], 'grad_z0': [], 'grad_theta': []}
    }

for meth in method_list:
    print("Processing method:", meth)
    for N in steps_list:

        baseline_sol, baseline_grad_z0, baseline_grad_theta = run_baseline(meth, N, device, t0, t1, dim)
        ac_sol, ac_grad_z0, ac_grad_theta = run_autocast_odeint(meth, N, device, t0, t1, dim, prec)
        manual_sol, manual_grad_z0, manual_grad_theta = pseudo_autocast(N, device, t0, t1, dim)
        print("ODEINT solution:", ac_sol, "gradient z0:", ac_grad_z0, "theta:", ac_grad_theta)
        print("Manual solution:", manual_sol, " z0:", manual_grad_z0, "Manual gradient  theta:", manual_grad_theta)
