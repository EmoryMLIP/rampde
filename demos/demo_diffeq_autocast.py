# ------------------------------------------------------------
# This script compares torchdiffeq.odeint under autocast 
# and manual “pseudo‐autocast” Euler integrator & backward. 
# ------------------------------------------------------------



import torch
import torch.nn as nn
from torch.amp import autocast

# Try to import torchdiffeq, gracefully handle if not available
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    print("Error: This demo requires torchdiffeq. Install with 'pip install torchdiffeq'")
    import sys
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != 'cuda':
    raise RuntimeError("Please run on a CUDA device")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prec = torch.float16

# Problem setup
dim = 8
t0, t1 = 0.0, 1.0
N = 64 
torch.manual_seed(42)


class ODEFunc(nn.Module):

    def __init__(self, dim, init_A=None):
        super().__init__()
        if init_A is None:
            A = torch.randn((dim, dim), dtype=torch.float32, device=device) 
        else:
            A = init_A.clone().to(device)
        self.theta = nn.Parameter(A)

    def forward(self, t, z):
        return torch.matmul(self.theta, z)


def run_autocast_odeint(func, N, device, t0, t1, dim, prec):
    """
    torchdiffeq.odeint under autocast
    """

    # print(f"run_autocast_odeint: N={N}, t0={t0}, t1={t1}, dim={dim}, prec={prec}")
    t_span = torch.linspace(t0, t1, N, device=device)
    z0 = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)
    with autocast(device_type='cuda', dtype=prec):

        sol = odeint(func, z0, t_span, method='euler')
        loss = 0.5 * ((sol[-1] - target)**2).sum()
        loss.backward()

    return sol.detach(), z0.grad.detach().clone(), func.theta.grad.detach().clone()


def manual_pseudoac(z_list, func, t0, t1, h, N, cast_matmul_inputs=False):
    """
    Backward pass with manual casting
    Returns:
      dL/dz0 (float32 Tensor), dL/dtheta (float32 Tensor)
    """
    N = len(z_list) -1
    # # print(f"manual_pseudoac: N={N}, len(z_list)={len(z_list)}, t0={t0}, t1={t1}, cast_matmul_inputs={cast_matmul_inputs}")
    # h = (t1 - t0) / (N)
    h = torch.tensor(h, dtype=torch.float32, device=z_list[0].device)
    print(f"manual_pseudoac: h={h}, t0={t0}, t1={t1}, N={N}")
    dim = func.theta.shape[0]
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=z_list[0].device)
    
    dL_dz = [None] * (N + 1)
    dL_dz[-1] = (z_list[-1] - target).to(torch.float32)
    
    dL_dtheta = torch.zeros_like(func.theta)
    
    for k in reversed(range(N)):


        # matmul for dL/dz
        dL_next = dL_dz[k + 1]                     # float32
        dL_part = torch.matmul((h*func.theta.to(prec)).transpose(0, 1), dL_next.to(prec))   
        dL_dz[k] = dL_next.to(torch.float32) + dL_part.to(torch.float32)           # float32
        
        # dL/dtheta
        dz = dL_dz[k + 1]#.to(prec)                          # float32
        z_prev = z_list[k]#.to(prec)                         # float32
        
        matmul_a = dz.to(prec)
        matmul_b = z_prev.transpose(0, 1).to(prec)
        
        d_theta_part = torch.matmul(h*matmul_a, matmul_b)  # float32
        dL_dtheta = dL_dtheta.to(prec)  + d_theta_part.to(prec)
        
    return dL_dz[0].to(torch.float32), dL_dtheta.to(torch.float32)


def euler_forward(z0, func, t0, t1, N, h):
    """
    Forward Euler manual implementation
    """
    
    # print(f"euler_forward: N={N}, h={h}, t0={t0}, t1={t1}")
    h = torch.tensor(h, dtype=torch.float32, device=z0.device)
    z_list = [z0.clone()]
    current_z = z0.clone()
    for _ in range(N - 1):
        f_val = torch.matmul(func.theta.to(prec), current_z.to(prec))  # float32
        z_next = current_z.to(torch.float32) + (h*f_val).to(torch.float32)             # float32
        z_list.append(z_next)
        current_z = z_next.to(torch.float32)  # ensure next z is float32
    return z_list


if __name__ == "__main__":
    h = (t1 - t0) / (N -1)

    init_A = torch.randn((dim, dim), dtype=torch.float32, device=device)*0.01
    func1 = ODEFunc(dim, init_A=init_A).to(device)
    func2 = ODEFunc(dim, init_A=init_A).to(device)


    sol_full, odeint_grad_z0, odeint_grad_theta = run_autocast_odeint(
        func1, N, device, t0, t1, dim, prec
    )


    z0_manual = torch.ones((dim, 1), dtype=torch.float32, device=device)
    z_list_manual = euler_forward(z0_manual, func2, t0, t1, N, h)
    options=False
    manual_grad_z0, manual_grad_theta = manual_pseudoac(z_list_manual, func2, t0, t1, h, N, cast_matmul_inputs=options)


    final_odeint = sol_full[-1]       # float32
    final_manual = z_list_manual[-1]  # float32
    diff_final = torch.abs(final_odeint - final_manual).max().item()

    gradz0_diff = torch.abs(odeint_grad_z0 - manual_grad_z0).max().item()
    gradtheta_diff = torch.abs(odeint_grad_theta - manual_grad_theta).max().item()

    print(f"Max abs‐difference in final z:        {diff_final:.6f}")      
    print(f"Max abs‐difference in dL/dz0:          {gradz0_diff:.6f}")      
    print(f"Max abs‐difference in dL/dtheta:           {gradtheta_diff:.6f}")  

