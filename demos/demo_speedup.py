#!/usr/bin/env python
# linear_ode_mixed_precision.py
#
# Accuracy & speed study for mixed-precision ODE integration with torchdiffeq
# Forward + backward, FP32 reference vs. FP16 compute (Tensor-Core)
#
# Author: <your name> – 2025-05-20
#
# ------------------------------------------------------------

import time, csv, os, re, math, sys
from contextlib import nullcontext

import torch
# from torchdiffeq import odeint
from rampde import odeint, NoScaler

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def _cuda_ms(evt):
    """Return self CUDA time in ms (PyTorch 1.13 and 2.x compatible)."""
    return getattr(evt, "self_cuda_time_total",
                   getattr(evt, "cuda_time_total", 0)) / 1e6


# ------------------------------------------------------------
# 1.  EXPERIMENT PARAMETERS
# ------------------------------------------------------------
device      = "cuda" if torch.cuda.is_available() else "cpu"

d           = 128 * 16          # feature dimension  (multiple of 8 ⇒ Tensor Cores)
B           = 512 * 32         # batch size
Nsteps      = 3         # fixed-step grid
t_final     = 1.0
methods     = ["euler", "rk4"]   # two solvers to test
dtype_low   = torch.float16
dtype_hi    = torch.float32

torch.manual_seed(0)

# ------------------------------------------------------------
# 2.  ANALYTIC SETUP  y' = A y,  A = -I  ⇒  y(t)=e^{-t}y0
# ------------------------------------------------------------
A  = -torch.eye(d, device=device, dtype=dtype_hi)

def analytic_solution(y0, t):
    """y(t) = e^{-t} * y0"""
    t = torch.as_tensor(t, device=y0.device, dtype=y0.dtype)
    return torch.exp(-t) * y0

def analytic_grad(y0, t):
    """∂(sum(y(t)))/∂y0 = e^{-t}  (same shape as y0)"""
    t = torch.as_tensor(t, device=y0.device, dtype=y0.dtype)
    return torch.full_like(y0, torch.exp(-t))

# ------------------------------------------------------------
# 3.  ODE MODULE
# ------------------------------------------------------------
class LinearODE(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
        d = A.shape[0]
        self.layer = torch.nn.Linear(d, d, bias=False,device = A.device)
        self.layer.weight.data = A

    def forward(self, t, y):
        return self.layer(y) 


# ------------------------------------------------------------
# 4.  DRIVER --------------------------------------------------
def run_one(method: str, mixed: bool):
    """
    method  ∈ {"euler","rk4"}
    mixed   = True  → FP16 compute + autocast
    """

    # --- choose dtypes & ODE rhs ---
    rhs      = LinearODE(A).to(device)
    y0       = torch.randn(B, d, device=device, requires_grad=True)   
    # --- time grid (high-precision is fine) ---
    t_grid   = torch.linspace(0.0, t_final, Nsteps + 1,
                              device=device, dtype=dtype_hi)

    # --- contexts for autocast / dtype ---
    ac_ctx   = torch.autocast(device_type="cuda", dtype=dtype_low) if mixed else nullcontext()

    # --------------------------------------------------
    # Forward timing

    # warmup run
    with ac_ctx:
        y_all = odeint(rhs, y0, t_grid, method=method)
        yN    = y_all[-1]
        loss = yN.to(dtype_hi).sum()
        loss.backward()
    y0.grad = None
    rhs.zero_grad()
    torch.cuda.synchronize()       
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fwd_start, fwd_end = torch.cuda.Event(True), torch.cuda.Event(True)

    with ac_ctx:
        fwd_start.record()        
        y_all = odeint(rhs, y0, t_grid, method=method)
        yN    = y_all[-1]
        fwd_end.record()
    torch.cuda.synchronize()
    fwd_time = fwd_start.elapsed_time(fwd_end) / 1e3   # seconds
    mem_fwd  = torch.cuda.max_memory_allocated() / 1e6 # MB

    # analytic reference
    y_true   = analytic_solution(y0.to(dtype_hi), t_final)
    y_err    = (yN.to(dtype_hi) - y_true).abs().max().item()

    # --------------------------------------------------
    # Backward timing  (loss = sum(yN))
    torch.cuda.reset_peak_memory_stats()
    bwd_start, bwd_end = torch.cuda.Event(True), torch.cuda.Event(True)

    loss = yN.to(dtype_hi).sum()         # keep loss in hi-precision
    bwd_start.record()
    loss.backward()
    bwd_end.record()
    torch.cuda.synchronize()
    bwd_time = bwd_start.elapsed_time(bwd_end) / 1e3
    mem_bwd  = torch.cuda.max_memory_allocated() / 1e6

    grad_true = analytic_grad(y0.detach().to(dtype_hi), t_final)
    grad_err  = (y0.grad.to(dtype_hi) - grad_true).abs().max().item()

    # --------------------------------------------------
    # PROFILER (backward)
    prof_bwd_csv, prof_bwd_txt = None, None
    fn_bwd_prefix = f"profile_bwd_{method}_{'mixed' if mixed else 'fp32'}"
    prof_bwd_csv  = fn_bwd_prefix + ".csv"
    prof_bwd_txt  = fn_bwd_prefix + ".txt"

    with ac_ctx:
        y_all = odeint(rhs, y0, t_grid, method=method)
        yN    = y_all[-1]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=True, profile_memory=False) as prof_bwd:
        with ac_ctx:
            loss_bwd = yN.to(dtype_hi).sum()
            loss_bwd.backward()

    # pretty txt
    with open(prof_bwd_txt, "w") as f:
        f.write(prof_bwd.key_averages().table(sort_by="self_cuda_time_total",
                                              row_limit=200, top_level_events_only=False))

    # csv
    header = ["name", "cuda_time_%", "cuda_time_ms"]
    rows_bwd, total_bwd = [], sum(_cuda_ms(e) for e in prof_bwd.key_averages())
    for e in prof_bwd.key_averages():
        rows_bwd.append([getattr(e, "name", getattr(e, "key", "unknown")),
                         100 * _cuda_ms(e) / (total_bwd or 1e-9),
                         _cuda_ms(e)])
    with open(prof_bwd_csv, "w", newline="") as fout:
        csv.writer(fout).writerows([header] + rows_bwd)

    # --------------------------------------------------
    # PROFILER (forward only, mixed run)
    prof_csv, prof_txt, tc_found, tc_time = None, None, None, None
    fn_prefix = f"profile_{method}_{'mixed' if mixed else 'fp32'}"
    prof_csv  = fn_prefix + ".csv"
    prof_txt  = fn_prefix + ".txt"

    def get_cuda_ms(evt):
        return getattr(evt, "self_cuda_time_total",
                        getattr(evt, "cuda_time_total", 0)) / 1e6

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False, with_stack=False) as prof:
        with ac_ctx:
            _ = odeint(rhs, y0, t_grid, method=method)

    # save pretty table
    with open(prof_txt, "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total",
                                            row_limit=200))

    # save CSV
    header = ["name", "cuda_time_%", "cuda_time_ms"]
    rows   = []
    total  = sum(get_cuda_ms(e) for e in prof.key_averages())
    for e in prof.key_averages():
        ms = get_cuda_ms(e)
        evt_name = getattr(e, "name", getattr(e, "key", "unknown"))
        rows.append([evt_name,
                        100 * ms / (total or 1e-9),
                        ms])
    with open(prof_csv, "w", newline="") as fout:
        csv.writer(fout).writerows([header] + rows)

    # Tensor-Core kernel search
    tc_pat = re.compile(r"(wmma|mma|tc)", re.IGNORECASE)
    tc_time = sum(r[2] for r in rows if tc_pat.search(r[0]))
    tc_found = tc_time > 0.0
    # --------------------------------------------------
    return dict(method=method, mixed=mixed,
                fwd_time=fwd_time, bwd_time=bwd_time,
                mem_fwd=mem_fwd, mem_bwd=mem_bwd,
                y_err=y_err, g_err=grad_err,
                prof_csv=prof_csv, prof_bwd_csv=prof_bwd_csv,
                tc_found=tc_found, tc_time=tc_time)


# ------------------------------------------------------------
# 5.  MAIN  ---------------------------------------------------
if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=True)

    results = []
    for meth in methods:
        results.append(run_one(meth, mixed=False))   # FP32 ref
        results.append(run_one(meth, mixed=True))    # MP16

    # Pretty print summary
    line = "-" * 72
    print("\n" + line)
    print(f"{'Method':<6} {'Mode':<4} |  y-err  |  g-err  | Fwd(s) | Bwd(s) | Mem(MB) | TC?  ")
    print(line)
    for r in results:
        tag = "MP16" if r["mixed"] else "FP32"
        mem_tot = max(r["mem_fwd"], r["mem_bwd"])
        print(f"{r['method']:<6} {tag:<4} | "
              f"{r['y_err']:.2e} | {r['g_err']:.2e} | "
              f"{r['fwd_time']:.3f} | {r['bwd_time']:.3f} | "
              f"{mem_tot:7.0f} | "
              f"{'yes' if r['tc_found'] else 'n/a ':>3} | "
              f"{r['prof_bwd_csv']}")

    # Speed-up & mem-ratio quick check
    for meth in methods:
        ref  = next(r for r in results if r["method"] == meth and not r["mixed"])
        mp16 = next(r for r in results if r["method"] == meth and r["mixed"])
        spd  = ref["fwd_time"] / mp16["fwd_time"]
        memr = mp16["mem_fwd"]  / ref["mem_fwd"]
        print(f"\n{meth.upper()}  speed-up (Fwd): {spd:.2f}×  | mem ratio: {memr:.2f}")

    print("\nProfiler tables saved to  *.txt / *.csv  files.")
    print("Look for kernel names containing 'wmma', 'mma', or 'tc' "
          "to confirm Tensor-Core execution.\n")