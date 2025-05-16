import sys
import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torchdiffeq import odeint
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint
import torch.optim as optim

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        stiff_matrix = torch.tensor([[-5000.0,    0.0],
                                     [   0.0 , -0.1]], dtype=torch.float32)
        self.theta = nn.Parameter(stiff_matrix)
    
    def forward(self, t, z):
        # z'(t) = theta * z(t).
        return torch.matmul(self.theta, z)

def euler_forward_manual_fp16(z0, func, t0, t1, N):
    h = (t1 - t0) / (N - 1)
    z_list = []
    current_z = z0.clone()
    t = t0
    z_list.append(current_z)
    for _ in range(N - 1):
        f_val = func.forward(t, current_z)
        current_z = current_z + h * f_val
        z_list.append(current_z)
        t += h
    return z_list

def euler_forward_manual_pseudoac(z0, func, t0, t1, N):
    h = (t1 - t0) / (N - 1)
    z_list = []
    current_z = z0.clone() 
    t = t0
    z_list.append(current_z)
    for _ in range(N - 1):
        z_half = current_z
        f_val_half = torch.matmul(func.theta.half(), z_half.half())
        h_half = torch.tensor(h, dtype=torch.float16, device=current_z.device)
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
    I = torch.eye(dim, dtype=torch.float32, device=z_list[0].device)
    
    dL_dz_half = [None]*(N+1)
    dL_dz_half[N] = z_list[-1] - target
    dL_dtheta_half = torch.zeros_like(func.theta)
    
    for k in reversed(range(N)):
        factor_half = ((h * func.theta).transpose(0,1)).half()
        dL_dz_half[k] = dL_dz_half[k+1].float() + torch.matmul(factor_half, dL_dz_half[k+1].half())
        dL_dtheta_half = (dL_dtheta_half.half() 
                          + (h * torch.matmul(dL_dz_half[k+1].float(),
                                              z_list[k].float().transpose(0,1))).half())
        dL_dtheta_half = dL_dtheta_half.float()
    return dL_dz_half[0], dL_dtheta_half


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0, t1 = 0.0, 2.0
    N = 1000
    dim = 2 
    
    t_span = torch.linspace(t0, t1, N, device=device)
    target = torch.full((dim, 1), 2.0, dtype=torch.float32, device=device)
    
    z0_auto = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    z0_autoac = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    z0_automp = torch.ones((dim, 1), dtype=torch.float32, device=device, requires_grad=True)
    func_auto = ODEFunc(dim).to(device)
    func_autoac = ODEFunc(dim).to(device)
    func_automp = ODEFunc(dim).to(device)
    
    # ==============================
    # 1) odeint full precision
    # ==============================
    sol_auto = odeint(func_auto, z0_auto, t_span, method="euler")
    L_auto = 0.5 * ((sol_auto[-1] - target)**2).sum()
    L_auto.backward()
    grad_z0_auto = z0_auto.grad.clone()
    grad_theta_auto = func_auto.theta.grad.clone()
    
    # ==============================
    # 2) odeint with autocast
    # ==============================
    with autocast(device_type='cuda', dtype=torch.float16):
        sol_autoac = odeint(func_autoac, z0_autoac, t_span, method="euler")
        L_autoac = 0.5 * ((sol_autoac[-1] - target)**2).sum()
        L_autoac.backward()
    grad_z0_autoac = z0_autoac.grad.clone()
    grad_theta_autoac = func_autoac.theta.grad.clone()

    # ==============================
    # 3) mpodeint with autocast
    # ==============================
    with autocast(device_type='cuda', dtype=torch.float16):
        z_list_automp = mpodeint(func_automp, z0_automp, t_span, method="euler")
        L_automp = 0.5 * ((z_list_automp[-1] - target)**2).sum()
        L_automp.backward()
    grad_z0_automp = z0_automp.grad.clone()
    grad_theta_automp = func_automp.theta.grad.clone()

    # ==============================
    # 4) Manual pseudo-autocast
    # ==============================
    z0_manual_noac = torch.ones((dim, 1), dtype=torch.float32, device=device)
    func_manual_noac = ODEFunc(dim).to(device)
    z_list_manual_noac = euler_forward_manual_pseudoac(z0_manual_noac, func_manual_noac, t0, t1, N)
    dL_dz0_manual_noac, dL_dtheta_manual_noac = manual_pseudoac(z_list_manual_noac, func_manual_noac, t0, t1)
    
    
    # ==============================
    # 5) Manual pseudo-autocast check
    # ==============================
    with autocast(device_type='cuda', dtype=torch.float16):
        z0_manual_ac = torch.ones((dim, 1), dtype=torch.float32, device=device)
        func_manual_ac = ODEFunc(dim).to(device)
        z_list_manual_ac = euler_forward_manual_pseudoac(z0_manual_ac, func_manual_ac, t0, t1, N)
        dL_dz0_manual_ac, dL_dtheta_manual_ac = manual_pseudoac(z_list_manual_ac, func_manual_ac, t0, t1)
    

    torch.set_printoptions(precision=8)
    print("=== odeint full precision ===")
    print("dL/dz0:", grad_z0_auto)
    print("dL/dtheta:\n", grad_theta_auto)
    print("=== odeint with autocast ===")
    print("dL/dz0:", grad_z0_autoac)
    print("dL/dtheta:\n", grad_theta_autoac)
    print("=== Manual pseudo-autocast ===")
    print("dL/dz0:", dL_dz0_manual_noac)
    print("dL/dtheta:\n", dL_dtheta_manual_noac)
    print("=== Manual pseudo-aaacast ===")
    print("dL/dz0:", dL_dz0_manual_ac)
    print("dL/dtheta:\n", dL_dtheta_manual_ac)
    print("=== mpodeint with autocast ===")
    print("dL/dz0:", grad_z0_automp)
    print("dL/dtheta:\n", grad_theta_automp)

    scaler = GradScaler()
    optimizer = optim.Adam(func_autoac.parameters(), lr=1e-3)
    optimizer1 = optim.Adam(func_autoac.parameters(), lr=1e-3)
    n_epochs = 2
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        
        with autocast(device_type="cuda", dtype=torch.float16):
            sol_autoac = odeint(func_autoac, z0_autoac, t_span, method="euler")
            loss = 0.5 * ((sol_autoac[-1] - target)**2).sum()
            
            sol_auto = odeint(func_auto, z0_auto, t_span, method="euler")
            L_auto = 0.5 * ((sol_auto[-1] - target)**2).sum()
            L_auto.backward()
            optimizer1.step()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        grad_z0_auto = z0_auto.grad.clone()
        grad_theta_auto = func_auto.theta.grad.clone()
        grad_z0_autoaca = z0_autoac.grad.clone()
        grad_theta_autoaca = func_autoac.theta.grad.clone()
        print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")
        print("=== after ===")
        print("dL/dz0 with scaling:", grad_z0_autoaca)
        print("dL/dtheta with:\n", grad_theta_autoaca)
        print("---")
        print("dL/dz0 without scaling:", grad_z0_auto)
        print("dL/dtheta without:\n", grad_theta_auto)
