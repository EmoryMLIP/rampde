import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.amp import custom_fwd

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/')))

from Phi import Phi


# --- Unittest for verifying trHess correctness --- #
class TestPhiHess(unittest.TestCase):
    def _run_device_test(self, device):
        torch.manual_seed(42)
        d = 2
        m = 5
        nTh = 2
        # Create an instance of Phi and move it to the target device.
        phi = Phi(nTh=nTh, m=m, d=d).to(device)
        phi.eval()
        # Manually set weights to fixed values for reproducibility.
        with torch.no_grad():
            phi.N.layers[0].weight.fill_(0.1)
            phi.N.layers[0].bias.fill_(0.2)
            phi.N.layers[1].weight.fill_(0.3)
            if hasattr(phi.N.layers[1], 'bias'):
                phi.N.layers[1].bias.fill_(0.0)
            phi.c.weight.fill_(0.0)
            phi.c.bias.fill_(0.0)
            phi.w.weight.fill_(1.0)
            phi.A.fill_(0.05)

        # Create an input batch (shape: [n_ex, d+1]).
        x = torch.randn(3, d + 1, device=device, requires_grad=True)

        # --- Compute gradients over the entire batch ---
        y = phi(x)  # shape (n_ex, 1)
        # Compute gradient d(phi)/dx for each sample (shape: [n_ex, d+1]).
        grad_autograd = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

        # --- Compute Hessian trace without an explicit loop using torch.vmap ---
        def compute_sample_hessian(phi, x_sample):
            # x_sample should have shape (d+1,)
            # Define a function that takes an input and returns the scalar output for that sample.
            f_single = lambda xi: phi(xi.unsqueeze(0))[0, 0]
            return torch.autograd.functional.hessian(f_single, x_sample)

        hess_trace_list = []
        for i in range(x.shape[0]):
            H = compute_sample_hessian(phi, x[i])
            trace_i = H[:-1,:-1].diagonal().sum()
            hess_trace_list.append(trace_i)
        hess_trace_autograd = torch.stack(hess_trace_list)
        # --- Get gradient and trace from phi.trHess ---
        grad_trhess, trace_trhess = phi.trHess(x)
        # Compare the gradients.
        self.assertTrue(
            torch.allclose(grad_autograd, grad_trhess, atol=1e-4, rtol=1e-3),
            f"Gradients differ on {device}:\nautograd:\n{grad_autograd}\ntrHess:\n{grad_trhess}"
        )
        print(f"Gradients pass on {device}:\nautograd:\n{grad_autograd}\ntrHess:\n{grad_trhess}")
        # Compare the Hessian traces.
        self.assertTrue(
            torch.allclose(hess_trace_autograd, trace_trhess, atol=1e-4, rtol=1e-3),
            f"Hessian traces differ on {device}:\nautograd:\n{hess_trace_autograd}\ntrHess:\n{trace_trhess}"
        )

        # --- Also test the justGrad option ---
        grad_only = phi.trHess(x, justGrad=True)
        self.assertTrue(
            torch.allclose(grad_autograd, grad_only, atol=1e-4, rtol=1e-3),
            f"justGrad gradients differ on {device}:\nautograd:\n{grad_autograd}\njustGrad:\n{grad_only}"
        )

    def test_trHess_cpu(self):
        self._run_device_test('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_trHess_gpu(self):
        self._run_device_test('cuda')


if __name__ == '__main__':
    unittest.main()