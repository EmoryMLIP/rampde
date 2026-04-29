#!/usr/bin/env python
"""Empirical stability probe for a trained parabolic CNN block.

Integrates y' = f(y) with forward Euler under six precision modes and
sweeps the step size around the Dahlquist boundary
    h* = tau_euler / rho(J),   tau_euler = 2,
similar to Croci & Rosilho de Souza (JCP 2022).

The block is the ode2 layer of the STL-10 network trained in paper/stl10.
That network is non-autonomous: K(t) and b(t) are learned as piecewise-
constant functions over four time intervals.  This probe freezes time at
the first interval (``--interval-idx 0`` by default) so the right-hand
side is autonomous, which the Dahlquist analysis assumes.

Two block variants are compared:

  linearized: f(y) = -K^T (M o K y) with the ReLU mask M = 1{K y0 + b > 0}
              frozen at y0.  The Jacobian J = -K^T diag(M) K is constant
              and symmetric NSD, so the dynamics reduce exactly to the
              Dahlquist test equation.
  parabolic : f(y) = -K^T ReLU(K y + b).  The Jacobian is still symmetric
              NSD (Croci & Rosilho de Souza, Assumption 3.2), but now
              nonlinear through the state-dependent ReLU mask.

Initial conditions y0 are real STL-10 test images pushed through the
trained opening layers
    stem -> norm1 -> ReLU -> ode1 (rk4, 4 steps) -> conn1 -> norm3 -> ReLU -> avg_pool2d
so the ode2 operator is probed on the activation distribution it actually
sees at inference.  The spectral radius of J is estimated on the same y0
via power iteration on Jacobian-vector products; the step size is set to
h = alpha * 2 / rho(J), with ``rho(J)`` taken as the maximum over the
batch (most conservative choice).

Precision modes:
  fp64 / fp32        reference paths, via rampde.odeint.
  rampde             rampde's shipping bf16 path: fp32 state accumulator,
                     right-hand side in bf16 under autocast.
  rampde-fp16        rampde's shipping fp16 path: fp32 accumulator,
                     right-hand side in fp16 under autocast with
                     DynamicScaler auto-installed by odeint.
  naive-mixed        state, weights, step size, update all in bf16.
  naive-mixed-fp16   analogous pure-fp16 baseline.
"""

import argparse
import copy
import csv
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import torchvision.transforms as T
from torchvision.datasets import STL10

from rampde import odeint


# ── Config ────────────────────────────────────────────────────────────────

# Hyphens are not LaTeX-friendly, so CSV column names replace them.
CSV_MODE = {
    "fp64": "fp64",
    "fp32": "fp32",
    "rampde": "rampde",
    "naive-mixed": "naive_mixed",
    "rampde-fp16": "rampde_fp16",
    "naive-mixed-fp16": "naive_mixed_fp16",
}

PRECISION_MODES = ["fp64", "fp32",
                   "rampde", "naive-mixed",
                   "rampde-fp16", "naive-mixed-fp16"]
TAU_EULER = 2.0  # stability boundary on the negative real axis

# Trajectory length per rampde.odeint call. Caps the memory of the
# materialized trajectory tensor when n_steps is in the thousands.
EULER_CHUNK = 100

BLOCK_VARIANTS = ("linearized", "parabolic")

# Default paths relative to this file.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

DEFAULT_CKPT = str(
    ROOT_DIR / "paper" / "stl10" / "raw_data" / "ode_stl10"
    / "stl10_float32_rampde_rk4_stable_stable_lr_0.05_nepochs_160"
      "_batch_size_16_width_128_seed25_20251001_195604"
    / "ckpt.pth"
)
DEFAULT_DATA_ROOT = str(SCRIPT_DIR.parent / "stl10" / ".data" / "stl10")

# STL-10 preprocessing, matching paper/stl10/ode_stl10.py::get_stl10_loaders.
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2241, 0.2210, 0.2239)
STL10_INPUT_SIZE = 128  # images are resized to 128x128 for Tensor Core alignment


# ── Block definitions ────────────────────────────────────────────────────

class ParabolicBlock(nn.Module):
    """f(y) = -K^T ReLU(K y + b).  Jacobian is symmetric NSD."""

    def __init__(self, K, b):
        super().__init__()
        self.register_buffer("K", K.clone())
        self.register_buffer("b", b.clone())

    def forward(self, t, y):
        z = F.conv2d(y, self.K, self.b, padding=1)
        z = F.relu(z)
        z = F.conv_transpose2d(z, self.K, padding=1)
        return -z


class LinearizedBlock(nn.Module):
    """f(y) = -K^T (mask o K y), with ReLU mask frozen at y0.

    J = -K^T diag(mask) K is constant and symmetric NSD, so the dynamics
    reduce exactly to the Dahlquist test equation.
    """

    def __init__(self, K, b, y0):
        super().__init__()
        self.register_buffer("K", K.clone())
        with torch.no_grad():
            mask = (F.conv2d(y0, K, b, padding=1) > 0).float()
        self.register_buffer("mask", mask)  # (B, C, H, W)

    def forward(self, t, y):
        z = F.conv2d(y, self.K, None, padding=1)
        z = self.mask * z
        z = F.conv_transpose2d(z, self.K, padding=1)
        return -z


class _ODE1Func(nn.Module):
    """Right-hand side of the trained ode1 block.

    Piecewise-constant K(t), b(t) over four intervals; each interval has
    its own InstanceNorm.  Mirrors paper/stl10/ode_stl10.py::ODEFunc with
    is_stable=True (the -K^T applies the same filter as the first conv).
    """

    def __init__(self, weight_bank, bias_bank):
        super().__init__()
        self.register_buffer("weight_bank", weight_bank)
        self.register_buffer("bias_bank", bias_bank)
        n = weight_bank.shape[0]
        ch = weight_bank.shape[1]
        self.norms = nn.ModuleList(
            [nn.InstanceNorm2d(ch, affine=False) for _ in range(n)]
        )
        self.n_intervals = n

    def forward(self, t, y):
        t_val = float(t.item() if torch.is_tensor(t) else t)
        idx = max(0, min(int(t_val * self.n_intervals), self.n_intervals - 1))
        W = self.weight_bank[idx]
        b = self.bias_bank[idx]
        z = F.conv2d(y, W, b, padding=1)
        z = F.relu(z)
        z = self.norms[idx](z)
        z = F.conv_transpose2d(z, W, padding=1)
        return -z


# ── STL-10 input pipeline ────────────────────────────────────────────────

def load_stl10_test_images(batch, seed, data_root, device):
    """Return ``batch`` STL-10 test images at the deployed 128x128 resolution."""
    tfm = T.Compose([
        T.Resize(STL10_INPUT_SIZE),
        T.CenterCrop(STL10_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])
    ds = STL10(root=data_root, split="test", download=False, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:batch].tolist()
    imgs = torch.stack([ds[i][0] for i in idx])
    return imgs.to(device)


def build_ode2_input(ckpt_path, imgs, device):
    """Push ``imgs`` through the trained layers that precede ode2.

    stem -> norm1 -> ReLU -> ode1 (rk4, 4 steps) -> conn1 -> norm3 ->
    ReLU -> avg_pool2d(stride=2).  With 128x128 inputs this returns a
    (batch, 256, 64, 64) tensor -- exactly what ode2 sees at inference.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]

    stem = nn.Conv2d(3, 128, 3, padding=1, bias=True).to(device)
    stem.weight.data.copy_(sd["stem.weight"])
    stem.bias.data.copy_(sd["stem.bias"])

    norm1 = nn.InstanceNorm2d(128, affine=True).to(device)
    norm1.weight.data.copy_(sd["norm1.weight"])
    norm1.bias.data.copy_(sd["norm1.bias"])

    conn1 = nn.Conv2d(128, 256, 1, padding=0, bias=True).to(device)
    conn1.weight.data.copy_(sd["conn1.weight"])
    conn1.bias.data.copy_(sd["conn1.bias"])

    norm3 = nn.InstanceNorm2d(256, affine=True).to(device)
    norm3.weight.data.copy_(sd["norm3.weight"])
    norm3.bias.data.copy_(sd["norm3.bias"])

    ode1 = _ODE1Func(
        sd["ode1.func.weight_bank"], sd["ode1.func.bias_bank"]
    ).to(device).eval()

    for m in (stem, norm1, conn1, norm3):
        m.eval()

    with torch.no_grad():
        x = stem(imgs)
        x = norm1(x)
        x = F.relu(x)
        t_grid = torch.linspace(0.0, 1.0, 5, device=device, dtype=torch.float32)
        xt = odeint(ode1, x, t_grid, method="rk4", loss_scaler=False)
        x = xt[-1]
        x = conn1(x)
        x = norm3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, stride=2)
    return x.contiguous()


def load_parabolic_block(ckpt_path, interval_idx, device):
    """Build the ParabolicBlock from the checkpoint at ``interval_idx``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    K = sd["ode2.func.weight_bank"][interval_idx]  # (C, C, 3, 3)
    b = sd["ode2.func.bias_bank"][interval_idx]    # (C,)
    block = ParabolicBlock(K, b).to(device).eval()
    return block, K.shape[0]


# ── Spectral radius ──────────────────────────────────────────────────────

def spectral_radius(block, y, n_iter=50):
    """Power iteration for rho(J) via Jacobian-vector products, per sample.

    Uses torch.autograd.functional.jvp so the same routine works for
    both block variants without a hand-written Jacobian formula.
    Eigenvalues of J are non-positive; returns rho = -lambda_min(J)
    with shape (B,).
    """
    block_d = copy.deepcopy(block).double()
    y64 = y.double()
    B = y64.shape[0]

    v = torch.randn_like(y64)
    v = v / v.flatten(1).norm(dim=1).view(B, 1, 1, 1).clamp(min=1e-30)

    rho = torch.zeros(B, dtype=torch.float64, device=y.device)
    for _ in range(n_iter):
        _, Jv = torch.autograd.functional.jvp(
            lambda x: block_d(None, x), (y64,), (v,)
        )
        rho = (v.flatten(1) * Jv.flatten(1)).sum(dim=1)
        nrm = Jv.flatten(1).norm(dim=1).view(B, 1, 1, 1).clamp(min=1e-30)
        v = Jv / nrm

    return -rho


# ── Forward Euler integration ────────────────────────────────────────────
#
# fp64, fp32, rampde, and rampde-fp16 all dispatch to rampde.odeint.  The
# only differences are the state dtype and whether the call is wrapped in
# autocast:
#
#   fp64        : y0.double(),  no autocast
#   fp32        : y0.float(),   no autocast
#   rampde      : y0.float(),   autocast(bf16)
#   rampde-fp16 : y0.float(),   autocast(fp16); odeint installs
#                 FixedGridODESolverDynamic + DynamicScaler.
#
# rampde's FixedGridODESolverUnscaled computes dy in the low dtype and
# updates y in the high dtype with autocast disabled.  naive-mixed and
# naive-mixed-fp16 are hand-coded baselines with no fp32 accumulator.

def integrate_euler(block, y0, dt, n_steps, mode):
    """Forward Euler integration; returns (norms, y_final).

    rampde.odeint materializes the full trajectory, which is prohibitive
    at large ``n_steps``.  The call is chunked into ``EULER_CHUNK`` steps
    at a time while reusing ``yt[-1]`` as the next starting state; each
    chunk still flows through rampde.odeint, so the fp32 accumulator is
    exercised identically to a single long call.
    """
    device = y0.device
    norms = np.empty(n_steps + 1)

    with torch.no_grad():
        if mode in ("naive-mixed", "naive-mixed-fp16"):
            low = torch.float16 if mode.endswith("fp16") else torch.bfloat16
            blk = copy.deepcopy(block).to(dtype=low, device=device)
            y = y0.to(dtype=low).contiguous()
            dt_t = torch.tensor(dt, dtype=low, device=device)
            norms[0] = y.double().norm().item()
            for n in range(n_steps):
                y = y + dt_t * blk(None, y)
                val = y.double().norm().item()
                norms[n + 1] = val
                if not np.isfinite(val):
                    norms[n + 2 :] = np.nan
                    return norms, y.float()
            return norms, y.float()

        dtype = torch.float64 if mode == "fp64" else torch.float32
        blk = copy.deepcopy(block).double().to(device) if mode == "fp64" else block
        y = y0.to(dtype).contiguous()
        norms[0] = y.double().norm().item()

        offset = 0
        while offset < n_steps:
            take = min(EULER_CHUNK, n_steps - offset)
            t = torch.arange(take + 1, dtype=dtype, device=device) * dt
            if mode == "rampde":
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    yt = odeint(blk, y, t, method="euler", loss_scaler=False)
            elif mode == "rampde-fp16":
                # Without loss_scaler=False, odeint auto-installs the
                # Dynamic solver variant with DynamicScaler.
                with autocast(device_type="cuda", dtype=torch.float16):
                    yt = odeint(blk, y, t, method="euler")
            else:
                yt = odeint(blk, y, t, method="euler", loss_scaler=False)

            chunk_norms = yt.double().flatten(1).norm(dim=1).cpu().numpy()
            norms[offset + 1 : offset + take + 1] = chunk_norms[1:]
            bad = ~np.isfinite(chunk_norms)
            if bad.any():
                # chunk_norms[0] is the chunk's entry state, finite by
                # invariant, so first_bad >= 1.
                first_bad = int(np.argmax(bad))
                norms[offset + first_bad :] = np.nan
                return norms, yt[first_bad - 1].float()
            y = yt[-1]
            offset += take

    return norms, y.float()


def gradient_quality(block, y0_single, dt, n_steps, mode):
    """Backprop dL/dy0 for L = 0.5 ||y^N||^2 through Euler integration."""
    device = y0_single.device
    if mode in ("naive-mixed", "naive-mixed-fp16"):
        low = torch.float16 if mode.endswith("fp16") else torch.bfloat16
        blk = copy.deepcopy(block).to(dtype=low, device=device)
        y0 = y0_single.to(dtype=low).clone().contiguous().requires_grad_(True)
        dt_t = torch.tensor(dt, dtype=low, device=device)
        y = y0
        for _ in range(n_steps):
            y = y + dt_t * blk(None, y)
    else:
        dtype = torch.float64 if mode == "fp64" else torch.float32
        blk = copy.deepcopy(block).double().to(device) if mode == "fp64" else block
        y0 = y0_single.to(dtype).clone().contiguous().requires_grad_(True)
        t = torch.arange(n_steps + 1, dtype=dtype, device=device) * dt
        if mode == "rampde":
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                yt = odeint(blk, y0, t, method="euler", loss_scaler=False)
        elif mode == "rampde-fp16":
            with autocast(device_type="cuda", dtype=torch.float16):
                yt = odeint(blk, y0, t, method="euler")
        else:
            yt = odeint(blk, y0, t, method="euler", loss_scaler=False)
        y = yt[-1]

    loss = 0.5 * y.float().pow(2).sum()
    loss.backward()
    return y0.grad.float()


def gradient_metrics(g, g_ref):
    """Angle (rad) and relative norm error between g and g_ref.

    Assumes g_ref is the fp64 backward through a stable Euler trajectory,
    so its norm is strictly positive; no epsilon guards are needed.
    """
    gf = g.flatten().double()
    gr = g_ref.flatten().double()
    cos = (gf @ gr) / (gf.norm() * gr.norm())
    angle = torch.acos(cos.clamp(-1, 1)).item()
    rel_err = abs(gf.norm().item() - gr.norm().item()) / gr.norm().item()
    return angle, rel_err


# ── Main experiment ──────────────────────────────────────────────────────

def run_experiment(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint : {args.ckpt_path}")
    print(f"STL-10 root: {args.data_root}")

    imgs = load_stl10_test_images(
        args.batch_size, args.seed, args.data_root, device
    )
    print(
        f"images     : {tuple(imgs.shape)}  "
        f"mean={imgs.mean():.3f} std={imgs.std():.3f}"
    )

    y0 = build_ode2_input(args.ckpt_path, imgs, device)
    norms_y0 = y0.flatten(1).norm(dim=1)
    print(
        f"y0 (ode2)  : {tuple(y0.shape)}  "
        f"||y0||=[{norms_y0.min():.2f}, {norms_y0.max():.2f}]"
    )

    parabolic, ch = load_parabolic_block(args.ckpt_path, args.interval_idx, device)
    linearized = LinearizedBlock(parabolic.K, parabolic.b, y0).to(device).eval()
    print(f"channels   : {ch}   interval: {args.interval_idx}")

    print(f"\nSpectral radius ({args.n_power_iter} power iterations, fp64 jvp)...")
    rho_l = spectral_radius(linearized, y0, args.n_power_iter)
    rho_p = spectral_radius(parabolic, y0, args.n_power_iter)

    rho_info = {}
    for label, rho in (("linearized", rho_l), ("parabolic", rho_p)):
        rho_info[label] = {
            "per_sample": rho.tolist(),
            "min": rho.min().item(),
            "median": rho.median().item(),
            "max": rho.max().item(),
        }
        print(
            f"  {label:11s}: min={rho.min():.4f}  "
            f"med={rho.median():.4f}  max={rho.max():.4f}"
        )

    stability = {}
    gradients = {}

    for vname, block, rho_vals in (
        ("linearized", linearized, rho_l),
        ("parabolic", parabolic, rho_p),
    ):
        rho_max = rho_vals.max().item()
        stability[vname] = {}
        gradients[vname] = {}

        for alpha in args.alphas:
            dt = alpha * TAU_EULER / rho_max
            akey = f"{alpha}"
            print(f"\n  [{vname}] alpha={alpha}  dt={dt:.6e}  rho_max={rho_max:.4f}")

            stability[vname][akey] = {"dt": dt, "results": {}}
            gradients[vname][akey] = {"dt": dt, "results": {}}

            for mode in args.precision_modes:
                t0 = time.time()
                norms, _ = integrate_euler(block, y0, dt, args.n_steps, mode)
                elapsed = time.time() - t0

                ratio = (
                    float(norms[-1] / norms[0])
                    if np.isfinite(norms[-1]) and norms[0] > 0
                    else float("inf")
                )
                blowup = not np.isfinite(norms[-1])

                stability[vname][akey]["results"][mode] = {
                    "final_ratio": ratio,
                    "blowup": blowup,
                    "elapsed_s": elapsed,
                }
                tag = "BLOWUP" if blowup else f"ratio={ratio:.6e}"
                print(f"    {mode:12s}: {tag}  ({elapsed:.1f}s)")

            # Run gradient sweep only when fp64 forward was stable, so
            # the gradient alpha range adapts per block.
            fp64_res = stability[vname][akey]["results"].get("fp64", {})
            fp64_stable = (
                "fp64" in args.precision_modes
                and not fp64_res.get("blowup", True)
                and np.isfinite(fp64_res.get("final_ratio", np.inf))
            )
            if fp64_stable and args.n_steps_grad > 0:
                y0s = y0[0:1]
                # LinearizedBlock's frozen mask has a batch dim that must
                # match y, so rebuild at batch size 1 for the gradient test.
                if vname == "linearized":
                    block_for_grad = LinearizedBlock(
                        parabolic.K, parabolic.b, y0s
                    ).to(device).eval()
                else:
                    block_for_grad = block
                try:
                    g_ref = gradient_quality(
                        block_for_grad, y0s, dt, args.n_steps_grad, "fp64"
                    )
                except torch.cuda.OutOfMemoryError:
                    print("    grad fp64: OOM -- skipping gradient tests")
                    continue

                for mode in [m for m in args.precision_modes if m != "fp64"]:
                    try:
                        g = gradient_quality(
                            block_for_grad, y0s, dt, args.n_steps_grad, mode
                        )
                        angle, rel_err = gradient_metrics(g, g_ref)
                        gradients[vname][akey]["results"][mode] = {
                            "angle_rad": angle,
                            "rel_norm_error": rel_err,
                        }
                        print(
                            f"    grad {mode:12s}: angle={angle:.4e} rad  "
                            f"rel_err={rel_err:.4e}"
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(f"    grad {mode:12s}: OOM -- skipped")
                        gradients[vname][akey]["results"][mode] = {
                            "angle_rad": None,
                            "rel_norm_error": None,
                            "error": "OOM",
                        }

    meta = {
        "checkpoint": args.ckpt_path,
        "data_root": args.data_root,
        "interval_idx": args.interval_idx,
        "channels": ch,
        "input_resolution": STL10_INPUT_SIZE,
        "ode2_spatial_size": int(y0.shape[-1]),
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_steps_grad": args.n_steps_grad,
        "n_power_iter": args.n_power_iter,
        "alphas": args.alphas,
        "tau_euler": TAU_EULER,
        "seed": args.seed,
        "gpu": (
            torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
        ),
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_out = {"metadata": meta, "spectral_radius": rho_info}
    meta_path = results_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    write_csvs(stability, gradients, results_dir, args.alphas, args.precision_modes)
    print(f"CSVs saved to     {results_dir}/stability_*.csv, gradient_*.csv")

    output = {
        "metadata": meta,
        "spectral_radius": rho_info,
        "stability_sweep": stability,
        "gradient_quality": gradients,
    }
    with open(results_dir / "stability_probe_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print_summary(output)
    return output


# ── CSV writers ──────────────────────────────────────────────────────────

def _write_table(path, header, rows):
    """Write ``rows`` to ``path`` as CSV with ``header``.

    Non-finite cells are written as ``nan`` so pgfplots skips them; the
    first column (alpha) is written with %g and the rest with %.6e.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            cells = []
            for i, v in enumerate(row):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    cells.append("nan")
                elif i == 0:
                    cells.append(f"{v:g}")
                else:
                    cells.append(f"{v:.6e}")
            w.writerow(cells)


def write_csvs(stability, gradients, results_dir, alphas, precision_modes):
    """Write stability + gradient CSVs, one file per (block, metric).

    Column schema (block in {linearized, parabolic}):
      stability_<block>.csv       alpha, fp64, fp32, rampde, naive_mixed, ...
      gradient_angle_<block>.csv  alpha, fp32, rampde, naive_mixed, ...
      gradient_relerr_<block>.csv alpha, fp32, rampde, naive_mixed, ...

    Blowup is encoded as NaN so stability curves end at the empirical
    boundary without a cosmetic cap.
    """
    forward_modes = [m for m in precision_modes if m in CSV_MODE]
    grad_modes = [m for m in forward_modes if m != "fp64"]
    akeys = sorted({f"{a}" for a in alphas}, key=float)

    for vname in BLOCK_VARIANTS:
        header = ["alpha"] + [CSV_MODE[m] for m in forward_modes]
        rows = []
        for akey in akeys:
            row = [float(akey)]
            res = stability.get(vname, {}).get(akey, {}).get("results", {})
            for m in forward_modes:
                r = res.get(m, {})
                val = r.get("final_ratio", float("nan"))
                if r.get("blowup"):
                    val = float("nan")
                row.append(val)
            rows.append(row)
        _write_table(results_dir / f"stability_{vname}.csv", header, rows)

        header_g = ["alpha"] + [CSV_MODE[m] for m in grad_modes]
        rows_angle, rows_rel = [], []
        for akey in akeys:
            res = gradients.get(vname, {}).get(akey, {}).get("results", {})
            row_a = [float(akey)]
            row_r = [float(akey)]
            for m in grad_modes:
                r = res.get(m, {})
                row_a.append(r.get("angle_rad", float("nan")))
                row_r.append(r.get("rel_norm_error", float("nan")))
            rows_angle.append(row_a)
            rows_rel.append(row_r)
        _write_table(results_dir / f"gradient_angle_{vname}.csv", header_g, rows_angle)
        _write_table(results_dir / f"gradient_relerr_{vname}.csv", header_g, rows_rel)


# ── Summary table ────────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 80)
    print("STABILITY PROBE SUMMARY")
    print("=" * 80)

    for vname in BLOCK_VARIANTS:
        rho = results["spectral_radius"][vname]
        print(f"\n--- {vname.upper()} block ---")
        print(
            f"rho(J): min={rho['min']:.4f}  "
            f"median={rho['median']:.4f}  max={rho['max']:.4f}"
        )

        sweep = results["stability_sweep"].get(vname, {})
        modes = PRECISION_MODES
        hdr = f"{'alpha':>6s} |" + "".join(f" {m:>14s} |" for m in modes)
        print(f"\n{hdr}")
        print("-" * len(hdr))
        for akey in sorted(sweep, key=float):
            row = f"{float(akey):6.2f} |"
            for m in modes:
                r = sweep[akey]["results"].get(m, {})
                if r.get("blowup"):
                    row += f" {'BLOWUP':>14s} |"
                else:
                    row += f" {r.get('final_ratio', float('nan')):>14.4e} |"
            print(row)

        grad = results["gradient_quality"].get(vname, {})
        grad_modes = [m for m in modes if m != "fp64"]
        has_grad = any(grad.get(a, {}).get("results") for a in grad)
        if has_grad:
            print("\nGradient angle (rad):")
            hdr2 = f"{'alpha':>6s} |" + "".join(f" {m:>14s} |" for m in grad_modes)
            print(hdr2)
            print("-" * len(hdr2))
            for akey in sorted(grad, key=float):
                res = grad[akey].get("results", {})
                if not res:
                    continue
                row = f"{float(akey):6.2f} |"
                for m in grad_modes:
                    a = res.get(m, {}).get("angle_rad")
                    row += f" {a:>14.4e} |" if a is not None else f" {'N/A':>14s} |"
                print(row)


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stability probe for a trained parabolic CNN block, with "
            "initial conditions drawn from real STL-10 test images."
        )
    )
    p.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--interval-idx", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--n-steps-grad", type=int, default=500)
    p.add_argument("--n-power-iter", type=int, default=50)
    p.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.5, 0.9, 0.99, 1.01, 1.1],
    )
    p.add_argument("--precision-modes", nargs="+", default=PRECISION_MODES)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--results-dir", default="raw_data")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
