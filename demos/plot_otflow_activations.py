import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib.pyplot as plt

# activation functions from Phi.py
def antiderivTanh(x, cast=True):
    if cast:
        dtype = x.dtype
        x = x.to(torch.float16)
    act =  torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))
    if cast:
        act = act.to(dtype)
    return act

def derivTanh(x, cast=True):
    if cast:
        dtype = x.dtype
        x = x.to(torch.float16)
    act =  1 - torch.tanh(x).pow(2)
    if cast:
        act = act.to(dtype)
    return act

def evaluate(fn, x, use_autocast=False, manual_fp16=False):
    """
    Evaluate fn(x) under different settings:
      - use_autocast: wrap call in autocast to float16 on CUDA
      - manual_fp16: cast input to float16, but no autocast
    Returns result as float32 tensor.
    """
    if manual_fp16:
        x_in = x.to(torch.float16)
        return fn(x_in, cast=True).to(torch.float32)
    elif use_autocast:
        with autocast(device_type='cuda', dtype=torch.float16):
            return fn(x,cast=False).to(torch.float32)
    else:
        return fn(x)  # assume x is float32

def main():
    # 1) let x be log space spanning float16 range
    # float16 range is roughly [2**-14, 2**15]
    x = torch.logspace(-14, 15, steps=300, device='cuda', dtype=torch.float32)

    # containers
    results = {}

    for name, fn in [('antiderivTanh', antiderivTanh),
                     ('derivTanh',      derivTanh)]:
        # 2) float32 baseline
        y_fp32 = evaluate(fn, x, use_autocast=False, manual_fp16=False)
        # 3) autocast→fp16
        y_auto = evaluate(fn, x, use_autocast=True,  manual_fp16=False)
        # 4) manual fp16
        y_man  = evaluate(fn, x, use_autocast=False, manual_fp16=True)

        results[name] = {
            'fp32': y_fp32.cpu(),
            'autocast': y_auto.cpu(),
            'manual_fp16': y_man.cpu()
        }

    
    # make plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for row, (name, res) in enumerate(results.items()):
        ax_val = axes[row, 0]
        ax_err = axes[row, 1]

        # plot values
        ax_val.plot(x.cpu().numpy(), res['fp32'], label='FP32')
        ax_val.plot(x.cpu().numpy(), res['autocast'], label='autocast fp16', ls='--')
        ax_val.plot(x.cpu().numpy(), res['manual_fp16'], label='manual fp16', ls=':')
        ax_val.set_xscale('log')
        ax_val.set_yscale('log')
        ax_val.set_title(f'{name} values')
        ax_val.legend(loc='best')

        # plot errors: difference to fp32
        err_auto = (res['autocast'] - res['fp32']).abs()
        err_man  = (res['manual_fp16'] - res['fp32']).abs()
        ax_err.plot(x.cpu().numpy(), err_auto, label='|autocast−fp32|')
        ax_err.plot(x.cpu().numpy(), err_man,  label='|manual−fp32|', ls=':')
        ax_err.set_xscale('log')
        ax_err.set_yscale('log')
        ax_err.set_title(f'{name} error vs FP32')
        ax_err.legend(loc='best')

        for ax in (ax_val, ax_err):
            ax.set_xlabel('x')
            ax.set_ylabel('y')

    # save
    fig.suptitle('Activation functions: FP32 vs FP16 (autocast/manual)')
    fig.savefig('activation_fp16_comparison.png', dpi=300)
    print("Saved figure to activation_fp16_comparison.png")

if __name__ == '__main__':
    main()