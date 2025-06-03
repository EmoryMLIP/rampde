import torch
from torch.amp import autocast
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchdiffeq import odeint #torchmpnode

#dy/dt = theta * y
class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([[2.0]]))
    def forward(self, t, y):
        print(f"Inside forward(): parameter theta.dtype = {self.theta.dtype}, y.dtype = {y.dtype}")
        return self.theta @ y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y0 = torch.tensor([[1.0]], dtype=torch.float32, device=device, requires_grad=True)
t = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
func = ODEFunc().to(device)


with autocast(device_type='cuda', dtype=torch.float16):
    sol = odeint(func, y0, t, method='euler')

    print(f"\nAfter odeint under autocast:")
    print(f"sol.dtype       = {sol.dtype}")
    print(f"func.theta.dtype = {func.theta.dtype}")
    print(f"y0.dtype         = {y0.dtype}")

    loss = sol[-1].sum()
    loss.backward()
    print(f"\nGradients after backward:")
    print(f"y0.grad.dtype     = {y0.grad.dtype}")
    print(f"func.theta.grad.dtype = {func.theta.grad.dtype}")#
