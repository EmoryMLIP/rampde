#This code demonstrate the benefit of dynamic scaler to prevent 
#1) overflow: Take Aincrease = 10, LossDecrease = 1
#2) undeflow: Take Aincrease = 1, LossDecrease = 1e-2 
#in gradients caused by low-precision operations

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchdiffeq')
parser.add_argument('--scaler', type=str, choices=['noscaler', 'dynamicscaler'], default='dynamicscaler')
parser.add_argument('--precision', type=str, choices=['float32', 'float16', 'bfloat16'], default='float16')
args = parser.parse_args()

precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]

if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from torchmpnode import odeint
    from torchmpnode import NoScaler, DynamicScaler
    scaler_map = {
        'noscaler': NoScaler(dtype_low=args.precision),
        'dynamicscaler': DynamicScaler(dtype_low=args.precision)
    }
    scaler = scaler_map[args.scaler]
    solver_kwargs = {'loss_scaler': scaler}
else:
    print("Using torchdiffeq")
    from torchdiffeq import odeint
    solver_kwargs = {}
    args.scaler = 'autocast_gradscaler'

device = torch.device("cuda")

torch.manual_seed(0)
dim = 3
A_true = torch.tensor([[-1.0, 0.0, 0.0],
                       [0.0, -5.0, 0.0],
                       [0.0, 0.0, -30.0]], device=device)
Aincrease = 1
LossDecrease = 1e-2


t = torch.linspace(0.0, 1, 100, device=device)
y0 = torch.randn(dim, device=device)*0.0001

with torch.no_grad():
    exp_At = torch.matrix_exp(A_true * t[-1])
    yT_true = exp_At @ y0

class LinearODE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, dim, device=device) * Aincrease)

    def forward(self, t, y):
        return self.A @ y

def train():
    model = LinearODE(dim).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    y0_train = y0.clone().requires_grad_(True)
    gscaler = GradScaler()
    for step in range(1, 21):
        opt.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            # solve ODE from t=0 to t=1
            yT_pred = odeint(model, y0_train, t, method='rk4', **solver_kwargs)[-1]
            # yT_pred = odeint(model, y0_train, t, method='rk4')[-1]  
            loss = torch.nn.functional.mse_loss(yT_pred, yT_true)*LossDecrease
            print(f"  [ step {step:02d} ] Loss={loss:.4e}")

            if args.odeint == 'torchmpnode':
                loss.backward()
                opt.step()
            else:
                
                gscaler.scale(loss).backward()
                gscaler.step(opt)
                gscaler.update()
                # scheduler.step()

        
        loss_val = loss.detach().item()
        if step % 1 == 0:
            print(f"step {step:02d} Loss={loss_val:.4e}")
            grad_A_num = model.A.grad.detach().float()

            A64 = model.A.detach().clone().double().requires_grad_(True)
            y0_64 = y0.clone().double().requires_grad_(True)
            exp_At = torch.matrix_exp(A64 * t[-1].double())
            yT64 = exp_At @ y0_64
            loss64 = torch.nn.functional.mse_loss(yT64, yT_true.double())*LossDecrease
            grad_y0_ana, grad_A_ana = torch.autograd.grad(loss64, (y0_64, A64))

            rel_err_A = torch.linalg.norm(grad_A_num - grad_A_ana.float()) \
                        / torch.linalg.norm(grad_A_ana.float())
            print(f"rel_errdLdA {args.odeint}{args.precision}{args.scaler}: {rel_err_A:.3e}") #

            grad_y0_num = y0_train.grad.detach().float() 
            grad_y0_ana = grad_y0_ana.float()
            rel_err_y0 = torch.linalg.norm(grad_y0_num - grad_y0_ana) \
                        / torch.linalg.norm(grad_y0_ana)
            print(f"rel_errdLdy0 with {args.odeint}{args.precision}{args.scaler}: {rel_err_y0:.3e}") #

    return True

if __name__ == "__main__":

    train()

