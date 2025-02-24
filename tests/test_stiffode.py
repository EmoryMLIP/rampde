
#mpnode seems to be more stable

import torch
import torch.nn as nn
import torch.optim as optim
import unittest
import matplotlib.pyplot as plt
import os, sys
import time
from torchdiffeq import odeint, odeint_adjoint
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint 
from torchdiffeq import odeint 

from torch.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.A = nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))  

    def forward(self, t, x):
        x = x.to(self.A.dtype)
        return torch.matmul(x, self.A.T)

def train_model(model, z0, t_span, target, device, use_autocast, ode_solver, num_epochs):

    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.1)
    scaler = GradScaler()
    losses = []
    stiffness_ratios = []
    eigenvalue_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        if use_autocast:
            with autocast("cuda", dtype=torch.float16):
                pred = ode_solver(model, z0, t_span, method="rk4")[-1]
        else:
            pred = ode_solver(model, z0, t_span, method="rk4")[-1]
        
        loss = ((pred.float() - target) ** 2).mean()
        losses.append(loss.item())

        if use_autocast:
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(retain_graph=True)
            optimizer.step()

        eigenvalues = torch.linalg.eigvals(model.A.detach()).cpu().numpy()
        eigenvalue_history.append(eigenvalues)
        abs_eigenvalues = torch.abs(torch.tensor(eigenvalues))
        stiffness_ratio = torch.max(abs_eigenvalues) / torch.min(abs_eigenvalues + 1e-10)
        stiffness_ratios.append(stiffness_ratio.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}, Stiffness Ratio = {stiffness_ratio.item()}")

    return losses, model.A.data.clone(), stiffness_ratios, eigenvalue_history

class TestNeuralODE(unittest.TestCase):
    def _run_training(self, device, training_mode, ode_solver):
        # A_true = torch.diag(torch.linspace(-5, 5, 8)).to(device)
        A_true = torch.diag(torch.tensor([-5000.0, -10.0, -0.1], dtype=torch.float32)).to(device)
        
        z0 = torch.randn(1, 3, device=device, dtype=torch.float32, requires_grad=True) 
        t_span = torch.linspace(0, 2, 1024, device=device)
        exp_A_true = torch.matrix_exp(A_true)
        target = torch.matmul(z0, exp_A_true.T)

        if training_mode == 'fp32':
            model = ODEFunc().to(device)
            use_autocast = False
        elif training_mode == 'autocast':
            model = ODEFunc().to(device)
            use_autocast = True
        # elif training_mode == 'manual_fp16':
        #     model = ODEFunc().to(device).half()
        #     use_autocast = False
        else:
            raise ValueError("Invalid training_mode.")

        losses, final_A, stiffness_ratios, eigenvalue_history = train_model(
            model, z0, t_span, target, device, use_autocast, ode_solver, num_epochs=300
        )
        
        return losses, final_A, target, stiffness_ratios, eigenvalue_history


    def _plot_results(self, stiffness_results, loss_results, labels):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        for i, (stiffness, label) in enumerate(zip(stiffness_results, labels)):
            axs[i].plot(stiffness, label=label)
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Stiffness Ratio")
            axs[i].set_yscale("log")
            axs[i].legend()
            axs[i].set_title(label)
        plt.tight_layout()
        plt.savefig("stiffness_tracking.png")

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        for i, (loss, label) in enumerate(zip(loss_results, labels)):
            axs[i].plot(loss, label=label, color='r')
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Loss")
            axs[i].legend()
            axs[i].set_title(label)
        plt.tight_layout()
        plt.savefig("loss_tracking.png")

    def test_stiffness_tracking(self):
        print("Running FP32 training with odeint...")
        losses_fp32_torchdiffeq, final_A_fp32_torchdiffeq, _, stiffness_fp32_torchdiffeq, _ = self._run_training(device, "fp32", odeint)

        print("Running FP32 training with mpodeint...")
        losses_fp32_mpodeint, final_A_fp32_mpodeint, _, stiffness_fp32_mpodeint, _ = self._run_training(device, "fp32", mpodeint)

        print("Running Autocast FP16 training with odeint...")
        losses_autocast_torchdiffeq, final_A_autocast_torchdiffeq, _, stiffness_autocast_torchdiffeq, _ = self._run_training(device, "autocast", odeint)

        print("Running Autocast FP16 training with mpodeint...")
        losses_autocast_mpodeint, final_A_autocast_mpodeint, _, stiffness_autocast_mpodeint, _ = self._run_training(device, "autocast", mpodeint)

        self._plot_results(
            [stiffness_fp32_torchdiffeq, stiffness_fp32_mpodeint, stiffness_autocast_torchdiffeq, stiffness_autocast_mpodeint], #
            [losses_fp32_torchdiffeq, losses_fp32_mpodeint,losses_autocast_torchdiffeq, losses_autocast_mpodeint], # 
            ["FP32 TorchDiffEq", "FP32 MPNode", "Autocast TorchDiffEq", "Autocast MPNode"] #
        )


if __name__ == '__main__':
    unittest.main()
