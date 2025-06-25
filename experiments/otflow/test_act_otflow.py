import torch

# --- Ground-truth functions (float64 precision) ---
def antideriv_true(x):
    x64 = x.to(torch.float64)
    return torch.abs(x64) + torch.log1p(torch.exp(-2.0 * torch.abs(x64)))

def deriv_true(x):
    x64 = x.to(torch.float64)
    return 1.0 - torch.tanh(x64).pow(2)

def antideriv_orig_log(x):
    # cast to float16, use log, cast back to float32
    x16 = x.to(torch.float16)
    act16 = torch.abs(x16) + torch.log(1.0 + torch.exp(-2.0 * torch.abs(x16)))
    return act16.to(torch.float32)

def antideriv_orig_log1p(x):
    # cast to float16, use log1p, cast back to float32
    x16 = x.to(torch.float16)
    act16 = torch.abs(x16) + torch.log1p(torch.exp(-2.0 * torch.abs(x16)))
    return act16.to(torch.float32)

def antideriv_f32_log(x):
    # all in float32, use log
    x32 = x.to(torch.float32)
    return torch.abs(x32) + torch.log(1.0 + torch.exp(-2.0 * torch.abs(x32)))

def antideriv_f32_log1p(x):
    # all in float32, use log1p
    x32 = x.to(torch.float32)
    return torch.abs(x32) + torch.log1p(torch.exp(-2.0 * torch.abs(x32)))

# --- Derivative variants (derivTanh) ---
def derivTanh_orig(x):
    # cast to float16, compute 1 - tanh^2, cast back
    x16 = x.to(torch.float16)
    act16 = 1.0 - torch.tanh(x16).pow(2)
    return act16.to(torch.float32)

def derivTanh_f32(x):
    # all in float32
    x32 = x.to(torch.float32)
    return 1.0 - torch.tanh(x32).pow(2)

# 1) base grid
base = torch.linspace(-50.0, 50.0, steps=1001, dtype=torch.float32)
# 2) subnormal / underflow, near smallest normal f16, normal region, f16 max/min, beyond f16 range / overflow
ext_mags = torch.tensor([1e-7, 1e-5, 1e-2, 1000.0, 65504.0, 70000.0], dtype=torch.float32)
exts    = torch.cat([ ext_mags, -ext_mags ])

x_test = torch.cat([base, exts])


# --- 1) Error metrics vs float64 ground truth ---
true_a = antideriv_true(x_test)
true_d = deriv_true(x_test)

antideriv_variants = {
    "orig_log":     antideriv_orig_log,
    "orig_log1p":   antideriv_orig_log1p,
    "f32_log":      antideriv_f32_log,
    "f32_log1p":    antideriv_f32_log1p,
}
deriv_variants = {
    "deriv_orig":   derivTanh_orig,
    "deriv_f32":    derivTanh_f32,
}

print("Antiderivative errors (vs float64):")
for name, fn in antideriv_variants.items():
    res = fn(x_test).to(torch.float64)
    err = (res - true_a).abs()
    print(f"  {name:12s}  max_err = {err.max():.3e},  mean_err = {err.mean():.3e}")

print("\nDerivative errors (vs float64):")
for name, fn in deriv_variants.items():
    res = fn(x_test).to(torch.float64)
    err = (res - true_d).abs()
    print(f"  {name:12s}  max_err = {err.max():.3e},  mean_err = {err.mean():.3e}")

# --- 2) Underflow check for derivTanh_orig ---
print("\nUnderflow detection for derivTanh_orig:")

x_fine = torch.linspace(0, 10, steps=100001, dtype=torch.float32)
d_orig = derivTanh_orig(x_fine)
d_true = derivTanh_f32(x_fine)

# find first index where d_orig == 0 but d_true > 0
mask = (d_orig == 0.0) & (d_true > 0.0)
if mask.any():
    idx = mask.nonzero(as_tuple=False)[0].item()
    thr = x_fine[idx].item()
    print(f"  derivTanh_orig underflows to 0 at x ≥ {thr:.4f}")
    print("  Nearest values around threshold:")
    for j in range(max(0, idx-3), idx+3):
        xv = x_fine[j].item()
        dvs = d_orig[j].item()
        dtv = d_true[j].item()
        print(f"    x={xv:.4f}  deriv_orig={dvs:.4e}  true_deriv={dtv:.4e}")
else:
    print("  No underflow to zero detected in [0,10].")
