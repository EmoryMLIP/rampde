import torch
import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint
from torchmpnode import Euler, RK4, FixedGridODESolver

class TestFixedGridODESolver(unittest.TestCase):

    def setUp(self):
        self.dtype = torch.float64
        self.device = torch.device('cpu')
        self.A = torch.randn(2, 2, dtype=self.dtype, device=self.device)
        self.A = - self.A @ self.A.T 
        class FuncModule(torch.nn.Module):
            def __init__(self, A):
                super(FuncModule, self).__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.linear.weight = torch.nn.Parameter(A)

            def forward(self, t, y):
                return self.linear(y)

        self.func = FuncModule(self.A)
        self.y0 = torch.tensor([1.0, 0.0], dtype=self.dtype, device=self.device)
        self.T = 20
        self.num_steps = 100
        self.t = torch.linspace(0, self.T, self.num_steps + 1, dtype=self.dtype, device=self.device)

    def analytical_solution(self, t):
        exp_At = torch.matrix_exp(self.A * t)
        return torch.matmul(exp_At, self.y0)
    
    def analytical_derivative(self,t):
        exp_At = torch.matrix_exp(self.A * t)
        return t*torch.matmul(exp_At, self.y0), exp_At
    
    

    def test_gradient_accuracy(self):
        solvers = [Euler(), RK4()]
        for solver in solvers:
            print(f"Testing gradient accuracy for {solver.name}")
            y0 = self.y0.clone().detach().requires_grad_(True)
            A = self.A.clone().detach().requires_grad_(True)
            self.num_steps = 64
            t = torch.linspace(0, self.T, self.num_steps + 1, dtype=self.dtype, device=self.device,requires_grad=True)
            yt = odeint(self.func, y0, t, method=solver.name, dtype_hi=self.dtype)
            loss = torch.sum(yt)
            loss.backward()        
            dy0 = y0.grad.clone().detach()
            dA = self.func.linear.weight.grad.clone().detach()
            gt = t.grad.clone().detach()

            vy0 =torch.randn_like(y0)
            dt = self.T/self.num_steps
            vt = torch.clip(0.2*dt*torch.randn_like(t), -0.6*dt, 0.6*dt)
            vt[0]=0
            vt[-1] = 0
            vA = torch.randn_like(self.func.linear.weight)
            vA = - vA @ vA.T
            grad = torch.sum(dy0 * vy0) + torch.sum(dA * vA) + torch.sum(gt * vt)
            previous_E0 = None
            previous_E1 = None
            pass_count_E0 = 0
            pass_count_E1 = 0

            for k in range(20):
                h = 2**(-k)
                y0_p = y0 + h * vy0
                A_p = A + h * vA
                t_p = t + h * vt
                # set weights in self.func.linear.wright to A_p
                self.func.linear.weight = torch.nn.Parameter(A_p)
                yt_p = odeint(self.func, y0_p, t_p, method=solver.name, dtype_hi=self.dtype)
                loss_p = torch.sum(yt_p)

                E0 = torch.norm(loss_p - loss).item()
                E1 = torch.norm(loss_p - loss - grad * h).item()
                
                if previous_E0 is not None:
                    observed_order_E0 = np.log2(previous_E0 / E0)
                    observed_order_E1 = 0 if E1==0 else np.log2(previous_E1 / E1)
                    if observed_order_E0 > 0.8:
                        pass_count_E0 += 1
                    if observed_order_E1 > 1.6:
                        pass_count_E1 += 1
                    print(f"h: {h:.2e}, E0: {E0:.2e}, E1: {E1:.2e}, Observed order E0: {observed_order_E0:.2f}, Observed order E1: {observed_order_E1:.2f}")
                else:
                    print(f"h: {h:.2e}, E0: {E0:.2e}, E1: {E1:.2e}")

                previous_E0 = E0
                previous_E1 = E1

            self.assertTrue(pass_count_E0 >= 5, "E0 did not display expected order one decay.")
            self.assertTrue(pass_count_E1 >= 5, "E1 did not display expected order two decay.")

    
        
if __name__ == '__main__':
    unittest.main()