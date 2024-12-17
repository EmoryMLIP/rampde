import unittest
import torch
from torchdiffeq import odeint as torch_odeint

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint


class TestODEintEquivalence(unittest.TestCase):

    def setUp(self):
        # Define a simple linear ODE: dx/dt = x
        class CoupledODEFunc(torch.nn.Module):
            def __init__(self,d=10):
                super(CoupledODEFunc, self).__init__()
                self.A = torch.nn.Parameter(torch.randn(d,d))
                self.B = torch.nn.Parameter(torch.randn(d,d))
                self.C = torch.nn.Parameter(torch.randn(d,d))
                self.D = torch.nn.Parameter(torch.randn(d,d))


            def forward(self, t, x):
                y,z = x
                dydt = self.A@y + self.B@z
                dzdt = self.C@y + self.D@z
                return (dydt,dzdt)

        d = 10
        self.func = CoupledODEFunc(d)
        self.x0 = (torch.ones(d), torch.randn(d))  # initial condition
        self.t = torch.linspace(0, .1, 10)
        self.method = 'rk4'

    def _test_on_device(self, device):
        # Move data and func to the device
        self.func.to(device)
        x0 = tuple(x.to(device) for x in self.x0)
        t = self.t.to(device)

        # Solve with torchdiffeq
        torch_solution = torch_odeint(self.func, x0, t, method='rk4')
        loss = sum(torch.norm(sol)**2 for sol in torch_solution)
        loss.backward()
        grad = torch.cat([p.grad.view(-1) for p in self.func.parameters()])
        for p in self.func.parameters():
            p.grad = None

        
        # Solve with custom odeint
        my_solution = mpodeint(self.func, x0, t, method='rk4')
        my_loss = sum(torch.norm(sol)**2 for sol in my_solution)
        my_loss.backward()
        my_grad = torch.cat([p.grad.view(-1) for p in self.func.parameters()])
        for p in self.func.parameters():
            p.grad = None
        
        # check that the tuples torch_solution and my_solution have same length and elements have same shape
        self.assertEqual(len(torch_solution), len(my_solution))
        for ts, ms in zip(torch_solution, my_solution):
            self.assertEqual(ts.shape, ms.shape)

        # turn torch_solution and my_solution into vectors
        torch_solution = torch.cat([sol.reshape(-1) for sol in torch_solution])
        my_solution = torch.cat([sol.reshape(-1) for sol in my_solution])
        
        # print last 20 elements in torhc_solution and my_soluton
        print(f"torch_solution: {torch_solution[-30:]}")
        print(f"my_solution: {my_solution[-30:]}")
        # print absolute and relative norm of solutions
        print(f"Absolute norm of solution difference: {torch.norm((torch_solution - my_solution)).item()}")
        print(f"Relative norm of solution difference: {torch.norm((torch_solution - my_solution)/torch.norm(torch_solution)).item()}")

        # Compare
        self.assertTrue(torch.allclose(my_solution, torch_solution, rtol=1e-4, atol=1e6),
                        "The solutions from torchmpnode and torchdiffeq differ more than expected.")
                
        print(torch.norm((grad - my_grad)).item())
        print(torch.norm((grad - my_grad)/torch.norm(grad)).item())
        self.assertTrue(torch.allclose(grad, my_grad, rtol=1e-5, atol=1e-5),
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