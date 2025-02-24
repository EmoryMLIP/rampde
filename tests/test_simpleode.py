import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from torchdiffeq import odeint
from torch.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------

# ODE dx/dt = w, so x(1) = x0 + w, Loss = (x0 + w - target)^2
# x0 = 1, x1 = target = 2, w = 1

# ---------------------------

# Autocast training
# Final loss: 0.0 Final w: 1.0000004768371582
# .Manual fp32 training
# Final loss: 0.0 Final w: 1.0000004768371582
# .Manual fp16 training, input fp32
# Final loss: 3.725290298461914e-09 Final w: 1.0

# ---------------------------

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.w = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
    def forward(self, t, x):
        return self.w.expand_as(x)

def train_model(model, z0, t_span, target, device, use_autocast, num_epochs=500):
    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.1) 
    scaler = GradScaler()
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        if use_autocast:
            with autocast("cuda", dtype=torch.float16):
                # Convert z0 and t_span to half inside the block.
                pred = odeint(model, z0, t_span, method="rk4")[-1]
        else:
            pred = odeint(model, z0, t_span, method="rk4")[-1]
        
        
        loss = ((pred.float() - target) ** 2).mean() 
        losses.append(loss.item())
        
        if use_autocast:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    return losses, model.w.item()


class TestNeuralODE(unittest.TestCase):
    def _run_training(self, device, training_mode):
        """
        training_mode: one of 'fp32', 'autocast', 'manual_fp16'
        For 'fp32': model fp32, z0 fp32
        For 'autocast': model fp32, z0 fp32, autocast
        For 'manual_fp16': model fp16, z0 fp32
        """
        target = torch.tensor([[2.0]], device=device, dtype=torch.float32)
        t_span = torch.linspace(0, 1, 256, device=device) 
        
        if training_mode == 'fp32':
            model = ODEFunc().to(device)
            z0 = torch.tensor([[1.0]], device=device, dtype=torch.float32, requires_grad=True)
            use_autocast = False
        elif training_mode == 'autocast':
            model = ODEFunc().to(device) 
            z0 = torch.tensor([[1.0]], device=device, dtype=torch.float32, requires_grad=True)
            use_autocast = True
        elif training_mode == 'manual_fp16':
            model = ODEFunc().to(device)
            model = model.half() 
            z0 = torch.tensor([[1.0]], device=device, dtype=torch.float32, requires_grad=True)
            use_autocast = False 
        else:
            raise ValueError("Invalid training_mode.")

        losses, final_w = train_model(model, z0, t_span, target, device, use_autocast, num_epochs=20)
        return losses, final_w

    def test_fp32(self):
        print("Manual fp32 training")
        losses, final_w = self._run_training(device, "manual_fp32")
        print("Final loss:", losses[-1], "Final w:", final_w)

    def test_autocast(self):
        print("Autocast training")
        losses, final_w = self._run_training(device, "autocast")
        print("Final loss:", losses[-1], "Final w:", final_w)

    def test_manual_fp16(self):
        print("Manual fp16 training")
        losses, final_w = self._run_training(device, "manual_fp16")
        print("Final loss:", losses[-1], "Final w:", final_w)

if __name__ == '__main__':
    unittest.main()
