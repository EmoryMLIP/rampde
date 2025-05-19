#!/usr/bin/env python
"""
Benchmark FP32 (+/– TF32), FP16-AMP, and BF16-AMP on Ampere/Hopper GPUs,
reporting avg time/iter **and peak GPU memory**.

PyTorch ≥ 2.1 and torchvision required.

A typical runtime on zuber

=== Mixed-Precision Benchmark (ResNet-50, batch 256) over 10 runs ===
Mode         |          ms/iter µ±σ |          peak MB µ±σ |             loss µ±σ
--------------------------------------------------------------------------------
fp32_tf32    |  660.91± 8.55     | 21399.0±  0.5     |  0.7667±0.3181
fp32_strict  |  880.10± 2.88     | 21399.2±  0.0     |  0.8848±0.3162
fp16_amp     |  366.86±16.05     | 11210.3±  0.0     |  0.8928±0.0000
bf16_amp     |  402.49±11.81     | 11210.3±  0.0     |  2.9351±0.0000

this demonstrates that TF32 
"""

import time, csv, random, pathlib, numpy as np
import torch, torchvision
from torch import nn, optim
from torch.amp import GradScaler, autocast


# ----------------------------------------------------------------------
#  Utility
# ----------------------------------------------------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_step(model, data, target, loss_fn, opt,
             scaler=None, use_amp=False, amp_dtype=torch.float16):
    opt.zero_grad(set_to_none=True)
    if use_amp:
        with autocast(device_type="cuda", dtype=amp_dtype):
            out = model(data)
            loss = loss_fn(out, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
    return loss.item()


def benchmark(mode_id: str,
              use_amp: bool,
              amp_dtype: torch.dtype | None,
              allow_tf32: bool,
              iters: int = 120,
              batch_size: int = 256):
    """
    Returns (avg_time_ms, last_loss, peak_mem_MB)
    """
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    device = "cuda"
    set_seed(0)

    model = torchvision.models.resnet50().to(device).train()
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().to(device)
    scaler  = GradScaler(enabled=use_amp)

    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    # warm-up
    for _ in range(10):
        one_step(model, x, y, loss_fn, opt,
                 scaler, use_amp, amp_dtype or torch.float32)
    torch.cuda.synchronize()

    # reset memory counters *after* warm-up
    torch.cuda.reset_peak_memory_stats(device)

    start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
    start_evt.record()
    for _ in range(iters):
        last_loss = one_step(model, x, y, loss_fn, opt,
                             scaler, use_amp, amp_dtype or torch.float32)
    end_evt.record()
    torch.cuda.synchronize()

    avg_ms   = start_evt.elapsed_time(end_evt) / iters
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    return avg_ms, last_loss, peak_mem


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as _np

    cfgs = [
        ("fp32_tf32",   False, None,           True),
        ("fp32_strict", False, None,           False),
        ("fp16_amp",    True,  torch.float16,  True),
        ("bf16_amp",    True,  torch.bfloat16, True),
    ]

    repeats = 10
    summary = []

    for mode_id, use_amp, amp_dtype, allow_tf32 in cfgs:
        t_list, loss_list, mem_list = [], [], []
        for run in range(repeats):
            t_ms, loss, mem = benchmark(mode_id, use_amp, amp_dtype, allow_tf32)
            t_list.append(t_ms)
            loss_list.append(loss)
            mem_list.append(mem)
        t_arr   = _np.array(t_list)
        loss_arr= _np.array(loss_list)
        mem_arr = _np.array(mem_list)

        summary.append({
            "mode": mode_id,
            "t_mean": t_arr.mean(),
            "t_std": t_arr.std(ddof=1),
            "loss_mean": loss_arr.mean(),
            "loss_std": loss_arr.std(ddof=1),
            "mem_mean": mem_arr.mean(),
            "mem_std": mem_arr.std(ddof=1),
        })

    # console summary
    print(f"\n=== Mixed-Precision Benchmark (ResNet-50, batch 256) over {repeats} runs ===")
    print(f"{'Mode':<12} | {'ms/iter µ±σ':>20} | {'peak MB µ±σ':>20} | {'loss µ±σ':>20}")
    print("-" * 80)
    for entry in summary:
        print(f"{entry['mode']:<12} | "
              f"{entry['t_mean']:7.2f}±{entry['t_std']:5.2f}     | "
              f"{entry['mem_mean']:7.1f}±{entry['mem_std']:5.1f}     | "
              f"{entry['loss_mean']:7.4f}±{entry['loss_std']:5.4f}")

    # optional CSV dump
    out = pathlib.Path("amp_benchmark_demo_summary.csv")
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "t_mean", "t_std", "mem_mean", "mem_std", "loss_mean", "loss_std"])
        for entry in summary:
            writer.writerow([
                entry["mode"],
                entry["t_mean"],
                entry["t_std"],
                entry["mem_mean"],
                entry["mem_std"],
                entry["loss_mean"],
                entry["loss_std"],
            ])
    print(f"\nResults saved to {out.resolve()}")