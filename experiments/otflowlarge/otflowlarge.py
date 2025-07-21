#adapted from https://github.com/EmoryMLIP/OT-Flow/blob/master/trainLargeOTflow.py
import os, sys
job_id = os.environ.get("SLURM_JOB_ID", "")
import argparse
import time
import random
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import time
import datetime
import pandas as pd

from torch.nn.functional import pad

# Add parent directory to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    setup_environment,
    get_precision_dtype,
    determine_scaler,
    setup_experiment
)

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser()
    
    # ODE solver arguments
    parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
    parser.add_argument('--precision', type=str, 
                        choices=['tfloat32', 'float32', 'float16','bfloat16'], default='float32',
                        help='Precision mode (float32 corresponds to --prec single in OT-Flow)')
    parser.add_argument('--odeint', type=str,
                        choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
    parser.add_argument('--adjoint', action='store_true')
    
    # Gradient scaling arguments
    parser.add_argument('--no_grad_scaler', action='store_true',
                        help='Disable GradScaler for torchdiffeq with float16 (default: enabled)')
    parser.add_argument('--no_dynamic_scaler', action='store_true',
                        help='Disable DynamicScaler for torchmpnode with float16 (default: enabled)')
    
    # Training arguments
    parser.add_argument('--niters', type=int, default=15000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_drop', type=float, default=10.0,
                        help='Factor to divide learning rate by (OT-Flow compatible)')
    parser.add_argument('--drop_freq', type=int, default=0,
                        help='Drop learning rate every N iterations (0=drop based on validation)')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Stop after this many validation checks without improvement')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping (run for full niters)')
    
    # Data arguments with OT-Flow aliases
    parser.add_argument('--data', type=str, default='miniboone',
                        choices=['miniboone', 'bsds300', 'power', 'gas', 'hepmass'],
                        help="Dataset to use")
    parser.add_argument('--batch_size', '--num_samples', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', '--num_samples_val', type=int, default=1024,
                        help='Validation/test batch size')
    
    # Model arguments with OT-Flow aliases
    parser.add_argument('--m', '--hidden_dim', type=int, default=256,
                        help='Hidden dimension of the model')
    parser.add_argument('--nt', '--num_timesteps', type=int, default=8,
                        help='Number of ODE solver timesteps for training')
    parser.add_argument('--nt_val', '--num_timesteps_val', type=int, default=None,
                        help='Number of ODE solver timesteps for validation (if None, uses nt)')
    parser.add_argument('--alph', '--alpha', type=str, default='1.0,100.0,15.0',
                        help='Alpha hyperparameters as comma-separated values (e.g., 1.0,100.0,15.0)')
    
    # Experiment arguments with OT-Flow aliases
    parser.add_argument('--val_freq', '--test_freq', type=int, default=100,
                        help='Validation frequency')
    parser.add_argument('--viz_freq', type=int, default=500,
                        help='Visualization frequency (not used in current implementation)')
    parser.add_argument('--results_dir', type=str, default="./results/otflowlarge")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    
    # Visualization
    parser.add_argument('--viz', action='store_true', default=True,
                        help="2D‐slice visualization on high-D data")
    
    return parser



def get_minibatch(X, num_samples):
    idx = torch.randint(0, X.size(0), (num_samples,), device=X.device)
    x = X[idx]
    B = x.size(0)
    z = torch.zeros(B, 1, dtype=torch.float32, device=X.device)
    return x, z.clone(), z.clone(), z.clone()

class OTFlow(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, alpha=[1.0]*2, Phi_class=None):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        if Phi_class is None:
            raise ValueError("Phi_class must be provided")
        self.Phi = Phi_class(2, hidden_dim, in_out_dim, alph=alpha)

    def forward(self, t, states):
        x = states[0]
        z = pad(x, (0,1,0,0), value=t)
        gradPhi, trH = self.Phi.trHess(z)
        dPhi_dx = gradPhi[:, :self.in_out_dim]
        dPhi_dt = gradPhi[:, self.in_out_dim].view(-1,1)

        dz_dt       = -(1.0/self.alpha[0]) * dPhi_dx
        dlogp_dt    = -(1.0/self.alpha[0]) * trH.view(-1,1)
        cost_L_dt   = 0.5 * torch.norm(dPhi_dx, dim=1, keepdim=True)**2
        cost_HJB_dt = torch.abs(-dPhi_dt + self.alpha[0]*cost_L_dt)
        return dz_dt, dlogp_dt, cost_L_dt, cost_HJB_dt

def load_data(name, datasets_module):
    """Load dataset using the datasets module passed from main()."""
    if name == 'bsds300':
        return datasets_module.BSDS300()
    elif name == 'power':
        return datasets_module.POWER()
    elif name == 'gas':
        return datasets_module.GAS()
    elif name == 'hepmass':
        return datasets_module.HEPMASS()
    elif name == 'miniboone':
        return datasets_module.MINIBOONE()
    else:
        raise ValueError('Unknown dataset')
        
def main():
    # Create parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set derived boolean values based on flags
    grad_scaler_enabled = not args.no_grad_scaler
    dynamic_scaler_enabled = not args.no_dynamic_scaler
    
    # Handle OT-Flow argument aliases
    # Use the primary names internally (batch_size, m, nt, nt_val, alph, val_freq)
    if hasattr(args, 'num_samples'):
        args.batch_size = args.num_samples
    if hasattr(args, 'num_samples_val'):
        args.test_batch_size = args.num_samples_val
    if hasattr(args, 'hidden_dim'):
        args.m = args.hidden_dim
    if hasattr(args, 'num_timesteps'):
        args.nt = args.num_timesteps
    if hasattr(args, 'num_timesteps_val'):
        args.nt_val = args.num_timesteps_val
    if hasattr(args, 'alpha'):
        args.alph = args.alpha
    if hasattr(args, 'test_freq'):
        args.val_freq = args.test_freq
    
    # Parse alpha hyperparameters
    args.alpha = [float(a) for a in args.alph.split(',')]
    
    # Set default for nt_val if not provided
    if args.nt_val is None:
        args.nt_val = args.nt
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups
    else:
        print("No seed specified, using random initialization")
    
    # Get base directory (torchmpnode root)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Setup environment and imports
    odeint_func, DynamicScaler = setup_environment(args.odeint, base_dir)
    
    # Handle adjoint method for torchdiffeq
    if args.odeint == 'torchdiffeq' and args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint_func
        print("Warning: Using torchdiffeq with adjoint method, which is not recommended for low precision training.")
    
    # Import utilities after setting up the path  
    from common import RunningAverageMeter, RunningMaximumMeter, AverageMeter
    from mmd import mmd
    import datasets
    from Phi import Phi
    
    # Get precision settings
    precision = get_precision_dtype(args.precision)
    
    # Determine scaler configuration
    loss_scaler, scaler_name, loss_scaler_for_odeint = determine_scaler(
        args.odeint, args.precision, grad_scaler_enabled, 
        dynamic_scaler_enabled, DynamicScaler
    )
    
    # Setup experiment directories and logging
    extra_params = {
        'alpha': args.alph,
        'lr': args.lr,
        'niters': args.niters,
        'batch_size': args.batch_size,
        'hidden_dim': args.m,
        'nt': args.nt
    }
    
    result_dir, ckpt_path, folder_name, device, log_file = setup_experiment(
        args.results_dir, "otflowlarge", args.data, args.precision,
        args.odeint, args.method, args.seed, args.gpu, scaler_name,
        extra_params=extra_params
    )
        
    # Copy the script to results directory
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))
    
    try:
        # load data
        data = load_data(args.data, datasets)
        train_x = torch.from_numpy(data.trn.x).float().to(device)
        val_x   = torch.from_numpy(data.val.x).float().to(device)
        d       = train_x.size(1)
        print(f"Loaded {args.data}: train={train_x.shape}, val={val_x.shape}")

        # setup model, optimizer, meters
        t0, t1 = 0.0, 1.0
        alpha  = args.alpha
        func   = OTFlow(in_out_dim=d, hidden_dim=args.m, alpha=alpha, Phi_class=Phi).to(device)
        optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cov = torch.eye(d, device=device) * 0.1
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d, device=device),
            covariance_matrix=cov
        )

        loss_meter     = RunningAverageMeter()
        NLL_meter      = RunningAverageMeter()
        cost_L_meter   = RunningAverageMeter()
        cost_HJB_meter = RunningAverageMeter()
        fwd_time_meter = AverageMeter()
        bwd_time_meter = AverageMeter()
        mem_meter      = RunningMaximumMeter()
        
        # CSV setup
        csv_path = os.path.join(result_dir, folder_name + ".csv")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "iter", "lr",
            "running_loss", "val_loss",
            "running_L",   "val_L",
            "running_NLL", "val_NLL",
            "running_HJB", "val_HJB",
            "time_fwd", "time_bwd", "time_fwd_sum", "time_bwd_sum", "max_memory_mb"
        ])

        # Check if a saved model exists and load it
        if os.path.exists(ckpt_path):
            cp = torch.load(ckpt_path, map_location=device)
            func.load_state_dict(cp['model_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            print(f"Loaded checkpoint {ckpt_path}")
        else:
            clampMax, clampMin = 1.5, -1.5  # OT-Flow compatible clamping
            
            # Early stopping variables (OT-Flow style)
            best_val_loss = float('inf')
            n_vals_wo_improve = 0
            ndecs = 0  # Number of learning rate decreases (OT-Flow compatible)
            
            itr = 1
            while itr <= args.niters:
                optimizer.zero_grad()
                z0, logp0, cL0, cH0 = get_minibatch(train_x, args.batch_size)
                torch.cuda.reset_peak_memory_stats(device)
                
                # clamp parameters
                for p in func.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)
                
                # Time forward pass
                torch.cuda.synchronize()
                fwd_start = time.perf_counter()

                with autocast(device_type='cuda', dtype=precision):
                    ts = torch.linspace(t0, t1, args.nt, device=device)
                    
                    # Prepare odeint arguments
                    odeint_kwargs = {'method': args.method}
                    if args.odeint == 'torchmpnode' and loss_scaler_for_odeint is not None:
                        odeint_kwargs['loss_scaler'] = loss_scaler_for_odeint
                    
                    # Single odeint call
                    z_t, logp_t, cL_t, cH_t = odeint_func(
                        func,
                        (z0, logp0, cL0, cH0),
                        ts,
                        **odeint_kwargs
                    )
                    z1, logp1, cL1, cH1 = z_t[-1], logp_t[-1], cL_t[-1], cH_t[-1]
                    logp_x = p_z0.log_prob(z1).view(-1,1) + logp1
                    loss   = (-alpha[2]*logp_x.mean()
                              + alpha[0]*cL1.mean()
                              + alpha[1]*cH1.mean())
                
                torch.cuda.synchronize()
                fwd_time = time.perf_counter() - fwd_start
                
                # Time backward pass
                torch.cuda.synchronize()
                bwd_start = time.perf_counter()
                
                # Handle backward pass with or without loss scaling
                if loss_scaler is not None and hasattr(loss_scaler, 'scale'):
                    # Track loss scale before step (for GradScaler only)
                    old_scale = loss_scaler.get_scale()
                    
                    # Use gradient scaling for torchdiffeq with float16
                    loss_scaler.scale(loss).backward()
                    
                    # Debug: Print gradient dtypes and check for inf/nan
                    if itr <= 5:  # Only for first few iterations
                        print(f"\n=== Iteration {itr} Gradient Debug ===")
                        print(f"{'Parameter':<25} {'Dtype':<10} {'Shape':<20} {'HasGrad':<8} {'IsFinite':<8} {'Min':<12} {'Max':<12}")
                        print("-" * 100)
                        for name, param in func.named_parameters():
                            if param.grad is not None:
                                is_finite = torch.isfinite(param.grad).all().item()
                                grad_min = param.grad.min().item() if is_finite else float('inf')
                                grad_max = param.grad.max().item() if is_finite else float('inf')
                                print(f"{name:<25} {str(param.grad.dtype):<10} {str(param.grad.shape):<20} {'Yes':<8} {str(is_finite):<8} {grad_min:<12.4e} {grad_max:<12.4e}")
                            else:
                                print(f"{name:<25} {'N/A':<10} {str(param.shape):<20} {'No':<8} {'N/A':<8} {'N/A':<12} {'N/A':<12}")
                        print("=" * 100)
                    
                    # Gradient clipping after backward pass
                    torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                    
                    # Track loss scale after step and log changes
                    new_scale = loss_scaler.get_scale()
                    if old_scale != new_scale:
                        print(f"Iteration {itr}: Loss scale changed from {old_scale} to {new_scale} (gradient overflow detected)")
                    elif itr < 20 or itr % 100 == 0:  # Log scale periodically for first 20 iterations or every 100
                        print(f"Iteration {itr}: Loss scale = {new_scale} (no overflow)")
                else:
                    # Standard backward pass (for DynamicScaler or no scaler)
                    loss.backward()
                    # Gradient clipping after backward pass
                    torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)
                    optimizer.step()
                
                torch.cuda.synchronize()
                bwd_time = time.perf_counter() - bwd_start
                
                peak_m  = torch.cuda.max_memory_allocated(device) / (1024**2)
                
                # Check for NaN or infinite loss
                if not torch.isfinite(loss).all():
                    print(f"Training stopped at iteration {itr}: Loss is {'NaN' if torch.isnan(loss).any() else 'infinite'}")
                    print(f"Loss value: {loss.item()}")
                    print("Saving current model state before stopping...")
                    torch.save({
                        'state_dict': func.state_dict(), 
                        'args': args,
                        'iteration': itr,
                        'loss': loss.item()
                    }, ckpt_path.replace('.pth', '_emergency_stop.pth'))
                    return  # Exit the training function
                
                # Check for NaN gradients (only if not using gradient scaling)
                if loss_scaler is None:
                    has_nan_grad = False
                    for name, param in func.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            print(f"Training stopped at iteration {itr}: NaN/infinite gradient detected in parameter '{name}'")
                            print(f"Gradient stats - min: {param.grad.min().item()}, max: {param.grad.max().item()}")
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        print("Saving current model state before stopping...")
                        torch.save({
                            'state_dict': func.state_dict(), 
                            'args': args,
                            'iteration': itr,
                            'loss': loss.item()
                        }, ckpt_path.replace('.pth', '_gradient_nan_stop.pth'))
                        return  # Exit the training function

                # update meters
                fwd_time_meter.update(fwd_time)
                bwd_time_meter.update(bwd_time)
                mem_meter.update(peak_m)
                loss_meter.update(loss.item())
                NLL_meter.update((-logp_x.mean()).item())
                cost_L_meter.update(cL1.mean().item())
                cost_HJB_meter.update(cH1.mean().item())

                # validation & CSV logging
                if itr % args.val_freq == 0:
                    with torch.no_grad():
                        vz0, vpl0, vL0, vH0 = get_minibatch(val_x, args.test_batch_size)
                        with autocast(device_type='cuda', dtype=precision):
                            # Validation: no scaler needed
                            vz_t, vpl_t, vL_t, vH_t = odeint_func(
                                func,
                                (vz0, vpl0, vL0, vH0),
                                torch.linspace(t0, t1, args.nt_val, device=device),
                                method=args.method
                            )
                            vz1, vpl1, vL1, vH1 = vz_t[-1], vpl_t[-1], vL_t[-1], vH_t[-1]
                            logp_val = p_z0.log_prob(vz1).view(-1,1) + vpl1
                            loss_val = (-alpha[2]*logp_val.mean()
                                        + alpha[0]*vL1.mean()
                                        + alpha[1]*vH1.mean())

                    print(f"[Iter {itr:5d}] LR {optimizer.param_groups[0]['lr']:.2e} | train loss {loss_meter.avg:.4f}, "
                          f"val loss {loss_val:.4f} | fwd {fwd_time_meter.avg:.3f}s, bwd {bwd_time_meter.avg:.3f}s | "
                          f"mem {mem_meter.max:.0f}MB")
                    
                    # Early stopping logic (only if not disabled)
                    if not args.no_early_stopping:
                        if loss_val.item() < best_val_loss:
                            best_val_loss = loss_val.item()
                            n_vals_wo_improve = 0
                            print(f"New best validation loss: {best_val_loss:.6f}")
                            # Save best model
                            torch.save({
                                'iteration': itr,
                                'model_state_dict': func.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_meter.avg,
                                'val_loss': best_val_loss,
                            }, ckpt_path)
                            print(f"Best model saved at {ckpt_path}")
                        else:
                            n_vals_wo_improve += 1
                            print(f"No improvement for {n_vals_wo_improve} validation checks")
                            
                            if n_vals_wo_improve >= args.early_stopping:
                                if ndecs >= 2:
                                    print(f"Early stopping engaged after {ndecs} LR reductions")
                                    break
                                else:
                                    # Reduce learning rate (OT-Flow style: divide by lr_drop)
                                    if ndecs == 0:
                                        new_lr = args.lr / args.lr_drop
                                    elif ndecs == 1:
                                        new_lr = args.lr / (args.lr_drop ** 2)
                                    else:
                                        new_lr = args.lr / (args.lr_drop ** (ndecs + 1))
                                    
                                    for g in optimizer.param_groups:
                                        g['lr'] = new_lr
                                    ndecs += 1
                                    print(f"Reduced LR to {optimizer.param_groups[0]['lr']:.2e} (reduction #{ndecs}, lr/{args.lr_drop}^{ndecs})")

                    # write one row to CSV
                    csv_writer.writerow([
                        itr,
                        optimizer.param_groups[0]['lr'],
                        loss_meter.avg,
                        loss_val.item(),
                        cost_L_meter.avg,
                        vL1.mean().item(),
                        NLL_meter.avg,
                        (-logp_val.mean()).item(),
                        cost_HJB_meter.avg,
                        vH1.mean().item(),
                        fwd_time_meter.avg,
                        bwd_time_meter.avg,
                        fwd_time_meter.sum,
                        bwd_time_meter.sum,
                        mem_meter.max
                    ])
                    csv_file.flush()

                # Periodic LR decay based on drop_freq (OT-Flow style)
                if args.drop_freq > 0 and itr % args.drop_freq == 0:
                    ndecs += 1
                    new_lr = args.lr / (args.lr_drop ** ndecs)
                    for g in optimizer.param_groups:
                        g['lr'] = new_lr
                    print(f"Periodic LR drop at iteration {itr}: new LR = {new_lr:.2e} (lr/{args.lr_drop}^{ndecs})")
                
                # Increment iteration counter for while loop
                itr += 1

        # Save final model
        final_model_path = ckpt_path.replace('.pth', '_final.pth')
        torch.save({
            'iteration': itr-1,
            'model_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_meter.avg,
        }, final_model_path)
        print(f"Final model saved at {final_model_path}")

    except KeyboardInterrupt:
        print("Interrupted. Exiting training loop.")
            
    finally:
        # Close CSV file if it was opened
        if 'csv_file' in locals() and csv_file:
            csv_file.close()

        # ------------------------------
        # Create optimization stats plots
        # ------------------------------
        if 'csv_path' in locals() and os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    data = np.array(list(reader), dtype=np.float32)

                if len(data) > 0:  # Only plot if we have data
                    iters       = data[:, 0]
                    lr_vals     = data[:, 1]
                    run_loss    = data[:, 2]
                    val_loss    = data[:, 3]
                    run_L       = data[:, 4]
                    val_L       = data[:, 5]
                    run_NLL     = data[:, 6]
                    val_NLL     = data[:, 7]
                    run_HJB     = data[:, 8]
                    val_HJB     = data[:, 9]
                    fwd_time    = data[:, 10]
                    bwd_time    = data[:, 11]
                    max_mem     = data[:, 12]

                    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

                    # 1) Loss
                    axs[0, 0].plot(iters, run_loss,    label="running loss")
                    axs[0, 0].plot(iters, val_loss,    label="val loss")
                    axs[0, 0].set_title("Loss Function")
                    axs[0, 0].set_xlabel("Iteration")
                    axs[0, 0].legend()

                    # 2) Transport cost L
                    axs[0, 1].plot(iters, run_L,    label="running L")
                    axs[0, 1].plot(iters, val_L,    label="val L")
                    axs[0, 1].set_title("Transport Cost L")
                    axs[0, 1].set_xlabel("Iteration")
                    axs[0, 1].legend()

                    # 3) NLL
                    axs[0, 2].plot(iters, run_NLL,  label="running NLL")
                    axs[0, 2].plot(iters, val_NLL,  label="val NLL")
                    axs[0, 2].set_title("Negative Log‐Likelihood")
                    axs[0, 2].set_xlabel("Iteration")
                    axs[0, 2].legend()

                    # 4) HJB penalty
                    axs[1, 0].plot(iters, run_HJB,  label="running HJB")
                    axs[1, 0].plot(iters, val_HJB,  label="val HJB")
                    axs[1, 0].set_title("HJB Penalty")
                    axs[1, 0].set_xlabel("Iteration")
                    axs[1, 0].legend()

                    # 5) Learning Rate
                    axs[1, 1].semilogy(iters, lr_vals, label="learning rate")
                    axs[1, 1].set_title("Learning Rate")
                    axs[1, 1].set_xlabel("Iteration")
                    axs[1, 1].legend()

                    # 6) Timing
                    axs[1, 2].plot(iters, fwd_time, label="forward time")
                    axs[1, 2].plot(iters, bwd_time, label="backward time")
                    axs[1, 2].set_title("Forward/Backward Pass Time")
                    axs[1, 2].set_xlabel("Iteration")
                    axs[1, 2].set_ylabel("Time (s)")
                    axs[1, 2].legend()

                    plt.tight_layout()
                    stats_fig = os.path.join(result_dir, "optimization_stats.png")
                    plt.savefig(stats_fig, bbox_inches='tight')
                    plt.close()
                    print(f"Saved optimization stats plot at {stats_fig}")

                    if args.viz and 'func' in locals() and 'val_x' in locals():
                        print("Generating 2D‐slice visualizations…")

                        val_np = val_x.cpu().numpy()
                        N      = min(val_np.shape[0], args.test_batch_size)
                        testData = val_np[:N]

                        #forward map f(x)
                        with torch.no_grad():
                            # x_batch = torch.from_numpy(testData).to(device)
                            vz0, vpl0, vL0, vH0 = get_minibatch(val_x, args.test_batch_size)
                            t_grid = torch.linspace(t0, t1, args.nt_val, device=device)
                            # Visualization: no scaler needed
                            z_t, logp_t, cL_t, cH_t   = odeint_func(
                                func,
                                (vz0, vpl0, vL0, vH0),
                                t_grid,
                                method=args.method
                            )
                        z_fwd = z_t[-1]
                        modelFx = z_fwd[:, :d].cpu().numpy()

                        #Gaussian samples & inverse map f⁻¹(y)
                        y = p_z0.sample([N]).to(device)

                        logp0 = torch.zeros(N, 1, device=device)
                        cL0   = torch.zeros_like(logp0)
                        cH0   = torch.zeros_like(logp0)
                        # for backward, reverse the time grid
                        t_grid_inv = torch.linspace(t1, t0, args.nt_val, device=device)
                        with torch.no_grad():
                            # Visualization: no scaler needed
                            z_inv_t, _, _, _ = odeint_func(
                                func,
                                (y, logp0, cL0, cH0),
                                t_grid_inv,
                                method=args.method
                            )
                        z_inv = z_inv_t[-1]
                        modelGen    = z_inv[:, :d].cpu().numpy()
                        normSamples = y.cpu().numpy()

                        nSamples = min(testData.shape[0], modelGen.shape[0])
                        testSamps  = testData[:nSamples, :]
                        modelSamps = modelGen[:nSamples, :]

                        mmd_val = mmd(modelSamps, testSamps)
                        print(f"MMD( ourGen , ρ0 )  num(ourGen)={modelSamps.shape[0]}, "
                              f"num(ρ0)={testSamps.shape[0]} : {mmd_val:.5e}")

                        nBins = 33
                        LOW, HIGH = -4, 4
                        if hasattr(data, 'gas') and result_dir.lower().find('gas')>=0:
                            LOW, HIGH = -2, 2
                        LOWrho0, HIGHrho0 = LOW, HIGH

                        bounds    = [[LOW, HIGH], [LOW, HIGH]]
                        boundsR0  = [[LOWrho0, HIGHrho0], [LOWrho0, HIGHrho0]]

                        for d1 in range(0, d-1, 2):
                            d2 = d1 + 1
                            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                            # fig.suptitle(f"Miniboone slices: dims {d1} vs {d2}", y=0.98, fontsize=18)

                            im1 = axs[0].hist2d(testData[:,d1], testData[:,d2],
                                 bins=nBins, range=boundsR0)[3]
                            axs[0].set_title(r"$x\sim\rho_0(x)$", fontsize=16)

                            # im2 = axs[0,1].hist2d(modelFx[:,d1], modelFx[:,d2],
                            #      bins=nBins, range=bounds)[3]
                            # axs[0,1].set_title(r"$f(x)$", fontsize=16)

                            # im3 = axs[1,0].hist2d(normSamples[:,d1], normSamples[:,d2],
                            #      bins=nBins, range=bounds)[3]
                            # axs[1,0].set_title(r"$y\sim\rho_1(y)$", fontsize=16)

                            im2 = axs[1].hist2d(modelGen[:,d1], modelGen[:,d2],
                                 bins=nBins, range=boundsR0)[3]
                            axs[1].set_title(r"$f^{-1}(y)$", fontsize=16)

                            for ax, im in zip(axs.flatten(), [im1, im2]): #im1,im2,im3,
                                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')

                            plt.tight_layout(rect=[0, 0, 1, 0.96])
                            out_file = os.path.join(result_dir,
                                f"slice_{d1}v{d2}.pdf")
                            plt.savefig(out_file, dpi=400)
                            plt.close(fig)

                        print(f"Saved visualizations in {result_dir}")
                    else:
                        print("Visualization skipped (use --viz).")
            except Exception as e:
                print(f"Error generating plots: {e}")
        # Close log file to restore stdout/stderr
        if 'log_file' in locals() and log_file:
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()

