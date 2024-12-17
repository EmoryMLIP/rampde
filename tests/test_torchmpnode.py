import unittest
import torch
from torchdiffeq import odeint as torch_odeint

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint


class TestODEintEquivalence(unittest.TestCase):

    def setUp(self):
        # Define a simple linear ODE: dx/dt = x
        class LinearODEFunc(torch.nn.Module):
            def __init__(self,d=10):
                super(LinearODEFunc, self).__init__()
                self.A = torch.nn.Parameter(torch.randn(d,d))
            def forward(self, t, x):
                return -(self.A.transpose(0,1) @ ( self.A@x))*1e-2
        d = 10
        self.func = LinearODEFunc(d)
        self.x0 = torch.ones(d)  # initial condition
        self.t = torch.linspace(0, 10, 100)
        self.method = 'rk4'

    def _test_on_device(self, device):
        # Move data and func to the device
        self.func.to(device)
        x0 = self.x0.to(device)
        t = self.t.to(device)


        # Solve with torchdiffeq
        torch_solution = torch_odeint(self.func, x0, t, method='rk4')
        torch.norm(torch_solution[-1]).backward()
        gradA = self.func.A.grad
        self.func.A.grad = None

        # Solve with custom odeint
        my_solution = mpodeint(self.func, x0, t, method='rk4')
        torch.norm(my_solution[-1]).backward()
        my_gradA = self.func.A.grad
        self.func.A.grad = None



        # print absolute error and relative error
        print(torch_solution[-1])
        print(my_solution[-1])
        print(torch.norm((torch_solution - my_solution)).item())
        print(torch.norm((torch_solution - my_solution)/torch.norm(torch_solution)).item())

        # print absolute and relative error of the gradients
        # print(gradA)
        # print(my_gradA)
        print(torch.norm((gradA - my_gradA)).item())
        print(torch.norm((gradA - my_gradA)/torch.norm(gradA)).item())

        # Compare
        self.assertTrue(torch.allclose(my_solution, torch_solution, rtol=1e-5, atol=1e-5),
                        "The solutions from torchmpnode and torchdiffeq differ more than expected.")
        
        self.assertTrue(torch.allclose(my_gradA, gradA, rtol=1e-5, atol=1e-5),
                        "The gradients from torchmpnode and torchdiffeq differ more than expected.")

    def test_on_cpu(self):
        device = torch.device('cpu')
        self._test_on_device(device)

    def test_on_cuda(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self._test_on_device(device)
        else:
            self.skipTest("CUDA not available")


if __name__ == '__main__':
    unittest.main()