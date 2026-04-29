"""Tests for stability_probe.py.

Three groups of checks:

1. ``spectral_radius`` matches a dense eigendecomposition of the Jacobian
   on both block variants.
2. The fp32 path of ``integrate_euler`` and ``gradient_quality`` reproduces
   a hand-coded fp32 Euler loop to tight tolerance.
3. ``naive-mixed`` is pure bf16 (state, weights, step size, update), in
   contrast to the fp32 state accumulator used by the order-preserving
   variants.
"""

import pathlib
import sys
import unittest

import torch
import torch.nn.functional as F

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stability_probe import (  # noqa: E402
    LinearizedBlock,
    ParabolicBlock,
    gradient_quality,
    integrate_euler,
    spectral_radius,
    TAU_EULER,
)


def _dense_jacobian(block, y_single):
    """Dense Jacobian of ``block`` at ``y_single`` of shape (1,C,H,W).

    Uses jvp with each canonical basis vector; returns an (n, n) matrix
    with n = C*H*W.
    """
    yb = y_single.double()
    flat_dim = yb.numel()
    cols = []
    for i in range(flat_dim):
        v_flat = torch.zeros(flat_dim, dtype=torch.float64, device=yb.device)
        v_flat[i] = 1.0
        v = v_flat.view_as(yb)
        _, Jv = torch.autograd.functional.jvp(
            lambda x: block(None, x), (yb,), (v,)
        )
        cols.append(Jv.flatten())
    return torch.stack(cols, dim=1)


class TestSpectralRadius(unittest.TestCase):
    """Cross-check spectral_radius against dense eigendecomposition."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C, self.H, self.W, self.B = 2, 4, 4, 2
        K = torch.randn(self.C, self.C, 3, 3, device=self.device) * 0.3
        b = torch.randn(self.C, device=self.device) * 0.1
        self.y = torch.randn(self.B, self.C, self.H, self.W, device=self.device) * 0.5
        self.parabolic = ParabolicBlock(K, b).to(self.device).eval()
        self.linearized = LinearizedBlock(K, b, self.y).to(self.device).eval()

    def _single_sample_block(self, block, b_idx):
        """Slice ``block`` to a single sample index.

        LinearizedBlock's mask carries a batch dim that would broadcast
        against a (1,C,H,W) probe vector and yield a non-square Jacobian;
        other blocks are unaffected.
        """
        if not isinstance(block, LinearizedBlock):
            return block.double()
        slim = LinearizedBlock.__new__(LinearizedBlock)
        torch.nn.Module.__init__(slim)
        slim.register_buffer("K", block.K.clone())
        slim.register_buffer("mask", block.mask[b_idx : b_idx + 1].clone())
        return slim.to(self.device).double()

    def _reference_rho(self, block):
        rhos = []
        for b_idx in range(self.B):
            blk_b = self._single_sample_block(block, b_idx)
            J = _dense_jacobian(blk_b, self.y[b_idx : b_idx + 1])
            eig = torch.linalg.eigvals(J)
            rhos.append(eig.abs().max().real.item())
        return torch.tensor(rhos, dtype=torch.float64, device=self.device)

    def _check(self, block, tol=1e-4):
        rho_ref = self._reference_rho(block)
        rho_pi = spectral_radius(block, self.y, n_iter=200)
        self.assertEqual(rho_pi.shape, (self.B,))
        self.assertTrue((rho_pi > 0).all(), f"rho must be positive, got {rho_pi}")
        rel = (rho_pi - rho_ref).abs() / rho_ref.clamp(min=1e-12)
        self.assertTrue(
            (rel < tol).all(),
            f"rho power-iter vs dense eig mismatch: pi={rho_pi}, ref={rho_ref}",
        )

    def test_linearized(self):
        self._check(self.linearized)

    def test_parabolic(self):
        self._check(self.parabolic)


class TestIntegrateEulerDelegation(unittest.TestCase):
    """rampde-backed fp32 path matches a reference Euler loop."""

    def setUp(self):
        torch.manual_seed(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C, H, W, B = 2, 4, 4, 1
        K = torch.randn(C, C, 3, 3, device=self.device) * 0.2
        b = torch.randn(C, device=self.device) * 0.1
        self.y0 = torch.randn(B, C, H, W, device=self.device) * 0.3
        self.block = ParabolicBlock(K, b).to(self.device).eval()
        rho = spectral_radius(self.block, self.y0, n_iter=100).max().item()
        self.dt = 0.5 * TAU_EULER / rho
        self.n_steps = 50

    def _reference_fp32(self):
        y = self.y0.float().clone()
        norms = [y.double().norm().item()]
        with torch.no_grad():
            for _ in range(self.n_steps):
                y = y + self.dt * self.block(None, y)
                norms.append(y.double().norm().item())
        return norms, y

    def test_fp32_matches_reference(self):
        norms_ref, y_ref = self._reference_fp32()
        norms, y_final = integrate_euler(
            self.block, self.y0, self.dt, self.n_steps, "fp32"
        )
        self.assertAlmostEqual(norms[-1], norms_ref[-1], delta=1e-4)
        self.assertLess((y_final - y_ref).norm().item(), 1e-3 * y_ref.norm().item())

    def test_fp64_stable_decays(self):
        norms, _ = integrate_euler(self.block, self.y0, self.dt, self.n_steps, "fp64")
        self.assertLess(norms[-1], norms[0])

    def test_rampde_runs(self):
        if not torch.cuda.is_available():
            self.skipTest("rampde requires CUDA autocast")
        norms, _ = integrate_euler(
            self.block, self.y0, self.dt, self.n_steps, "rampde"
        )
        self.assertTrue(norms[-1] > 0 and norms[-1] < 10 * norms[0])


class TestNaiveMixedExplicit(unittest.TestCase):
    """Semantics of the naive-mixed variant.

    All of state, weights, step size, and update are kept in bf16; the
    order-preserving variants instead accumulate the state in fp32.
    """

    def setUp(self):
        torch.manual_seed(2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.skipTest("naive-mixed requires CUDA (bfloat16)")
        C, H, W, B = 2, 4, 4, 1
        K = torch.randn(C, C, 3, 3, device=self.device) * 0.2
        b = torch.randn(C, device=self.device) * 0.1
        self.y0 = torch.randn(B, C, H, W, device=self.device) * 0.3
        self.block = ParabolicBlock(K, b).to(self.device).eval()
        self.rho = spectral_radius(self.block, self.y0, n_iter=100).max().item()

    def test_naive_mixed_final_is_bf16(self):
        """Every block call in naive-mixed receives a bf16 input.

        ``integrate_euler`` casts ``y_final`` to fp32 on return, so the
        dtype is verified via a wrapper that inspects the input on each
        forward call.
        """
        class AssertBF16(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.saw_non_bf16 = False

            def forward(self, t, y):
                if y.dtype != torch.bfloat16:
                    self.saw_non_bf16 = True
                return self.inner(t, y)

        wrapped = AssertBF16(self.block).to(self.device)
        dt = 0.5 * TAU_EULER / self.rho
        _ = integrate_euler(wrapped, self.y0, dt, n_steps=5, mode="naive-mixed")
        self.assertFalse(
            wrapped.saw_non_bf16,
            "naive-mixed must feed bf16 into the block at every step",
        )

    def test_naive_mixed_blows_up_earlier_than_rampde(self):
        """On the linearized block near the Dahlquist boundary, rampde
        is at least as accurate as naive-mixed against the fp64 reference."""
        lin = LinearizedBlock(self.block.K, self.block.b, self.y0).to(self.device).eval()
        rho = spectral_radius(lin, self.y0, n_iter=200).max().item()
        dt = 0.95 * TAU_EULER / rho
        n_steps = 500

        norms_ref, _ = integrate_euler(lin, self.y0, dt, n_steps, "fp64")
        norms_op, _ = integrate_euler(lin, self.y0, dt, n_steps, "rampde")
        norms_nv, _ = integrate_euler(lin, self.y0, dt, n_steps, "naive-mixed")

        err_op = abs(norms_op[-1] - norms_ref[-1]) / norms_ref[-1]
        err_nv = abs(norms_nv[-1] - norms_ref[-1]) / norms_ref[-1]
        self.assertLessEqual(err_op, err_nv + 1e-6)


class TestGradientQualityDelegation(unittest.TestCase):
    """fp32 backward through rampde matches a hand-coded backward."""

    def setUp(self):
        torch.manual_seed(3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C, H, W = 2, 4, 4
        K = torch.randn(C, C, 3, 3, device=self.device) * 0.2
        b = torch.randn(C, device=self.device) * 0.1
        self.y0 = torch.randn(1, C, H, W, device=self.device) * 0.3
        self.block = ParabolicBlock(K, b).to(self.device).eval()
        rho = spectral_radius(self.block, self.y0, n_iter=100).max().item()
        self.dt = 0.5 * TAU_EULER / rho
        self.n_steps = 20

    def _reference_grad_fp32(self):
        y = self.y0.float().clone().requires_grad_(True)
        z = y
        for _ in range(self.n_steps):
            z = z + self.dt * self.block(None, z)
        loss = 0.5 * z.float().pow(2).sum()
        loss.backward()
        return y.grad.float()

    def test_fp32_gradient_matches_reference(self):
        g_ref = self._reference_grad_fp32()
        g = gradient_quality(self.block, self.y0, self.dt, self.n_steps, "fp32")
        rel = (g - g_ref).norm().item() / g_ref.norm().item()
        self.assertLess(rel, 1e-3)


if __name__ == "__main__":
    unittest.main()
