#!/usr/bin/env python3
import os, sys
job_id = os.environ.get("SLURM_JOB_ID", "")
import argparse
import logging
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import datetime
import csv
import shutil
import glob
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import toy_data

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser()
    
    # Data and visualization
    parser.add_argument(
        '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
        type=str, default='swissroll')
    parser.add_argument('--viz', default=True, action='store_true')
    
    # Training parameters
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--test_freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--num_samples_val', type=int, default=1024)
    parser.add_argument('--num_timesteps', type=int, default=128)
    
    # Model parameters
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=32)
    
    # Method and precision
    parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
    parser.add_argument('--precision', type=str, choices=['tfloat32', 'float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
    parser.add_argument('--scaler', type=str, choices=['noscaler', 'dynamicscaler'], default='dynamicscaler')
    
    # Legacy arguments for compatibility
    parser.add_argument('--adjoint', action='store_true')
    
    # Experiment setup
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser

def setup_environment(args):
    """Setup the environment and imports based on args."""
    # Set up path for utils import
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, os.path.join(base_dir, "examples"))
    
    if args.odeint == 'torchmpnode':
        print("Using torchmpnode")
        assert args.method == 'rk4' 
        sys.path.insert(0, base_dir)
        from torchmpnode import odeint, NoScaler, DynamicScaler
        scaler_map = {
            'noscaler': NoScaler,
            'dynamicscaler': DynamicScaler
        }
        return odeint, scaler_map[args.scaler]
    else:    
        print("using torchdiffeq")
        try:
            if args.adjoint:
                from torchdiffeq import odeint_adjoint as odeint
            else:
                from torchdiffeq import odeint
            return odeint, None
        except ImportError:
            print("torchdiffeq not available, falling back to torchmpnode")
            sys.path.insert(0, base_dir)
            from torchmpnode import odeint, NoScaler, DynamicScaler
            scaler_map = {
                'noscaler': NoScaler,
                'dynamicscaler': DynamicScaler
            }
            return odeint, scaler_map.get(args.scaler, DynamicScaler)

def get_precision_dtype(precision_str):
    """Convert precision string to torch dtype."""
    precision_map = {
        'float32': torch.float32,
        'tfloat32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    return precision_map[precision_str]

def setup_precision(precision_str):
    """Setup precision-related settings."""
    if precision_str == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("Using strict float32 precision")
    elif precision_str == 'tfloat32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Using TF32 precision")

def setup_experiment(args, base_dir):
    """Setup experiment directories, logging, and environment."""
    job_id = os.environ.get("SLURM_JOB_ID", "")
    
    os.makedirs(args.results_dir, exist_ok=True)
    seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.data}_{args.precision}_{args.odeint}_{args.method}_nt_{args.num_timesteps}_{seed_str}_{timestamp}"
    result_dir = os.path.join(base_dir, "results", "cnf", folder_name)
    ckpt_path = os.path.join(result_dir, 'ckpt.pth')
    os.makedirs(result_dir, exist_ok=True)
    
    with open("result_dir.txt", "w") as f:
        f.write(result_dir)
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))

    # Save arguments to CSV file for easy loading
    args_csv_path = os.path.join(result_dir, "args.csv")
    args_dict = vars(args)
    args_df = pd.DataFrame([args_dict])
    args_df.to_csv(args_csv_path, index=False)

    # Redirect stdout and stderr to a log file.
    log_path = os.path.join(result_dir, folder_name + ".txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Setup precision
    setup_precision(args.precision)

    try:
        torch.backends.cudnn.verbose = True
    except:
        pass  # Some PyTorch versions don't have this attribute
    torch.backends.cudnn.benchmark = True

    # Print environment and hardware info for reproducibility and debugging
    print("Environment Info:")
    print(f"  Python version: {sys.version}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    try:
        print(f"  CUDA version: {torch.version.cuda}")
    except:
        print(f"  CUDA version: N/A")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print("   cuDNN enabled:", torch.backends.cudnn.enabled)
    print(f"  GPU Device Name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")
    print(f"  Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

    print("Experiment started at", datetime.datetime.now())
    print("Arguments:", vars(args))
    print("Results will be saved in:", result_dir)
    print("SLURM job id", job_id)
    print("Model checkpoint path:", ckpt_path)
    
    return result_dir, ckpt_path, folder_name, device, log_file


def hyper_trace(W, B, U, x, target_dtype):
    """Compute the trace of the Jacobian using the Hutchinson estimator."""
    W = W.to(target_dtype)  # [w, d, 1]
    B = B #.to(target_dtype)  # [w, 1, 1]
    U = U.to(target_dtype)  # [w, 1, d]
    x = x.to(target_dtype)  # [n, d]

    w, d, _ = W.shape
    n = x.shape[0]
    x_exp = x.unsqueeze(0).expand(w, -1, -1)  # [w, n, d]

    # s_j = x @ w_j + b_j
    s = torch.bmm(x_exp, W).squeeze(-1)       # [w, n]
    s = s.to(torch.float32) + B.to(torch.float32).squeeze(-1)                     # [w, n]
    deriv = 1 - torch.tanh(s.to(target_dtype))**2              # [w, n]

    # u_j * w_j 
    uw_dot = torch.bmm(U, W).squeeze(-1).squeeze(-1)  # [w]
    uw_dot = uw_dot.view(w, 1)                        # [w, 1]

    trace_all = deriv.to(target_dtype) * uw_dot       # [w, n]
    trace_sum = trace_all.to(torch.float32).sum(dim=0) 
    trace_est = trace_sum / w        
    trace_est = trace_est.to(target_dtype)  
    return trace_est.view(n, 1)  

class CNF(nn.Module):
    """Continuous Normalizing Flow.
    
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)  

    def forward(self, t, states):
        z = states[0]
        W, B, U = self.hyper_net(t) 

        z = z.to(W.dtype)
        Z = z.unsqueeze(0).repeat(self.width, 1, 1)
        h = torch.tanh(torch.matmul(Z, W) + B)
        f = torch.matmul(h, U).mean(0) 

        target_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
        trace_est = hyper_trace(W, B, U, z, target_dtype)  
        dlogp_z_dt = -trace_est
        dz_dt = f

        return (dz_dt, dlogp_z_dt)

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()
    
class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        blocksize = width * in_out_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        params = t.reshape(1, 1)
        params = params.to(self.fc1.weight.dtype)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)
        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]

def get_batch(num_samples, data_type, device):
    """Generate a batch of data points from the specified distribution."""
    points = toy_data.inf_train_gen(data_type, batch_size=num_samples)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    return x, logp_diff_t1

def compute_mmd_loss(samples1, samples2):
    """Compute MMD loss between two sample sets."""
    # Import MMD here to avoid circular imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
    from mmd import mmd
    
    # Convert to numpy if needed and compute MMD
    if isinstance(samples1, torch.Tensor):
        samples1 = samples1.detach().cpu().numpy()
    if isinstance(samples2, torch.Tensor):
        samples2 = samples2.detach().cpu().numpy()
    
    mmd_value = mmd(samples1, samples2, indepth=False, alph=1.0)
    return mmd_value

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop."""
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Create parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
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
    
    # Setup environment and imports
    odeint_func, scaler_class = setup_environment(args)
    
    # Import utilities after setting up the path
    from utils import RunningAverageMeter, RunningMaximumMeter
    
    # Get precision settings
    precision = get_precision_dtype(args.precision)
    
    # Get base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Setup experiment directories and logging
    result_dir, ckpt_path, folder_name, device, log_file = setup_experiment(args, base_dir)
    
    try:
        # Constants for the experiment
        viz_samples = 30000
        viz_timesteps = 41
        t0, t1 = 0.0, 1.0
        
        # Create model
        func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
        print(func)
        print('Number of parameters: {}'.format(count_parameters(func)))
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(func.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

        # Setup prior distribution
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0], device=device),
            covariance_matrix=torch.tensor([[1, 0.0], [0.0, 1]], device=device)
        )

        # Setup meters for tracking progress
        loss_meter = RunningAverageMeter()
        time_meter = RunningAverageMeter()
        mem_meter = RunningMaximumMeter()

        # Setup solver kwargs and loss scaling
        if args.odeint == 'torchmpnode' and scaler_class is not None:
            scaler = scaler_class(dtype_low=precision)
            solver_kwargs = {'loss_scaler': scaler}
            loss_scaler = None  # torchmpnode handles its own scaling
        else:
            solver_kwargs = {}
            # Use PyTorch's GradScaler for torchdiffeq with float16
            if args.precision == 'float16':
                from torch.cuda.amp import GradScaler
                loss_scaler = GradScaler()
                print("Using PyTorch GradScaler for float16 precision with torchdiffeq")
            else:
                loss_scaler = None

        # Setup CSV logging
        csv_path = os.path.join(result_dir, folder_name + ".csv")
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow([
            'iter', 'lr', 'running_NLL', 'val_NLL', 'val_NLL_mp', 
            'val_mmd', 'train_mmd', 'time_avg', 'max_memory'
        ])

        # Check for existing checkpoints
        checkpoint_files = glob.glob(os.path.join(result_dir, f'{args.odeint}_*.pth'))
        start_iter = 1
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            cp = torch.load(latest_checkpoint, map_location=device)
            func.load_state_dict(cp['func_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            start_iter = cp.get('iteration', 1) + 1
            print(f"Loaded checkpoint from {latest_checkpoint}, resuming from iteration {start_iter}")

        # Training loop
        for itr in range(start_iter, args.niters + 1):
            optimizer.zero_grad()
            
            # Get training batch
            x, logp_diff_t1 = get_batch(args.num_samples, args.data, device)
            
            # Start timing with synchronization
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()

            with autocast(device_type='cuda', dtype=precision):
                ts = torch.linspace(t1, t0, args.num_timesteps, device=device)
                z_t, logp_diff_t = odeint_func(
                    func,
                    (x, logp_diff_t1),
                    ts,
                    method=args.method,
                    **solver_kwargs
                )
                z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
                logp_x = p_z0.log_prob(z_t0) - logp_diff_t0.view(-1)
                loss = -logp_x.mean(0)

            # Handle backward pass with optional loss scaling
            if loss_scaler is not None:
                # Scale loss and backward pass for float16 with torchdiffeq
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                # Standard backward pass
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            # End timing with synchronization
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            
            # Update meters
            time_meter.update(elapsed_time)
            mem_meter.update(peak_memory)
            loss_meter.update(loss.item())

            # Check for NaN or infinite loss
            if not torch.isfinite(loss).all():
                print(f"Training stopped at iteration {itr}: Loss is {'NaN' if torch.isnan(loss).any() else 'infinite'}")
                print(f"Loss value: {loss.item()}")
                print("Saving current model state before stopping...")
                torch.save({
                    'func_state_dict': func.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': itr,
                    'loss': loss.item()
                }, ckpt_path.replace('.pth', '_emergency_stop.pth'))
                break

            # Evaluate and log every test_freq iterations
            if itr % args.test_freq == 0:
                # Compute validation losses
                with torch.no_grad():
                    # Validation loss (standard precision)
                    x_val, lpv1 = get_batch(args.num_samples_val, args.data, device)
                    ts_val = torch.linspace(t1, t0, args.num_timesteps, device=device)
                    z_v, lp_v = odeint_func(func, (x_val, lpv1), ts_val,
                                        atol=1e-5, rtol=1e-5,
                                        method=args.method,
                                        **solver_kwargs)
                    z0_v, lp0_v = z_v[-1], lp_v[-1]
                    logp_val = p_z0.log_prob(z0_v) - lp0_v.view(-1)
                    loss_val = -logp_val.mean()

                    # Validation loss (mixed precision)
                    with autocast(device_type='cuda', dtype=precision):
                        lpv1_mp = torch.zeros_like(lpv1)
                        z_v_mp, lp_v_mp = odeint_func(func, (x_val, lpv1_mp), ts_val,
                                                    method=args.method,
                                                    **solver_kwargs)
                        z0_mp, lp0_mp = z_v_mp[-1], lp_v_mp[-1]
                        logp_val_mp = p_z0.log_prob(z0_mp) - lp0_mp.view(-1)
                        loss_val_mp = -logp_val_mp.mean()

                    # Compute MMD metrics
                    # Generate target samples for comparison
                    target_samples, _ = get_batch(args.num_samples_val, args.data, device)
                    
                    # Generate model samples (forward pass from prior)
                    z_t0_sample = p_z0.sample([args.num_samples_val]).to(device)
                    logp_diff_t0_sample = torch.zeros(args.num_samples_val, 1, device=device, dtype=torch.float32)
                    ts_samples = torch.linspace(t0, t1, args.num_timesteps).to(device)
                    
                    with autocast(device_type='cuda', dtype=precision):
                        z_t_samples, _ = odeint_func(
                            func,
                            (z_t0_sample, logp_diff_t0_sample),
                            ts_samples,
                            method=args.method,
                            **solver_kwargs
                        )
                    generated_samples = z_t_samples[-1]
                    
                    # Compute MMD between target and generated samples
                    val_mmd = compute_mmd_loss(target_samples, generated_samples)
                    
                    # Compute MMD between training and validation target
                    train_target, _ = get_batch(args.num_samples, args.data, device)
                    train_mmd = compute_mmd_loss(train_target, target_samples)

                print(
                    f"Iter {itr:4d} | "
                    f"Train NLL: {loss_meter.avg:.4f} | "
                    f"Val NLL: {loss_val.item():.4f} | "
                    f"Val NLL (mp): {loss_val_mp.item():.4f} | "
                    f"Val MMD: {val_mmd:.4f} | "
                    f"Train MMD: {train_mmd:.4f} | "
                    f"Time avg: {time_meter.avg:.4f}s | "
                    f"Mem peak: {mem_meter.val:.2f}MB"
                )

                # Write to CSV
                writer.writerow([
                    itr,
                    optimizer.param_groups[0]['lr'],
                    loss_meter.avg,                 # running_NLL (loss is -logp)
                    loss_val.item(),                # val_NLL
                    loss_val_mp.item(),             # val_NLL_mp
                    val_mmd,
                    train_mmd,
                    time_meter.avg,
                    mem_meter.val
                ])
                csv_file.flush()

        # Save final checkpoint
        torch.save({
            'func_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': args.niters
        }, ckpt_path)
        
        csv_file.close()

        # Create optimization stats plots
        create_optimization_plots(csv_path, result_dir)
        
        # Create MMD plots
        create_mmd_plots(csv_path, result_dir)

        # Generate visualizations if requested
        if args.viz:
            generate_visualizations(func, p_z0, args, result_dir, device, precision, 
                                   odeint_func, solver_kwargs, viz_samples, viz_timesteps, t0, t1)

    finally:
        # Close log file to restore stdout/stderr
        if 'log_file' in locals() and log_file:
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

def create_optimization_plots(csv_path, result_dir):
    """Create optimization statistics plots."""
    df = pd.read_csv(csv_path)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 1) NLL subplot
    axs[0, 0].plot(df['iter'], df['running_NLL'], label="running NLL")
    axs[0, 0].plot(df['iter'], df['val_NLL'], label="val NLL")
    axs[0, 0].plot(df['iter'], df['val_NLL_mp'], "--", label="val NLL (mp)")
    axs[0, 0].set_title("Negative Log-Likelihood")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("NLL")
    axs[0, 0].legend()

    # 2) MMD subplot
    axs[0, 1].plot(df['iter'], df['val_mmd'], label="val MMD")
    axs[0, 1].plot(df['iter'], df['train_mmd'], label="train MMD")
    axs[0, 1].set_title("Maximum Mean Discrepancy")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("MMD")
    axs[0, 1].legend()

    # 3) Learning Rate subplot
    axs[1, 0].semilogy(df['iter'], df['lr'], label="learning rate")
    axs[1, 0].set_title("Learning Rate")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("LR")
    axs[1, 0].legend()

    # 4) Max Memory subplot
    axs[1, 1].plot(df['iter'], df['max_memory'], label="max memory (MB)")
    axs[1, 1].set_title("Max Memory")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Memory (MB)")
    axs[1, 1].legend()

    plt.tight_layout()
    stats_fig_path = os.path.join(result_dir, "optimization_stats.png")
    plt.savefig(stats_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved optimization stats plot at {stats_fig_path}")

def create_mmd_plots(csv_path, result_dir):
    """Create MMD-specific plots."""
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['iter'], df['val_mmd'], label='Validation MMD', marker='o')
    plt.plot(df['iter'], df['train_mmd'], label='Train MMD', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('MMD')
    plt.title('Maximum Mean Discrepancy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    mmd_plot_path = os.path.join(result_dir, "mmd_plot.png")
    plt.savefig(mmd_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved MMD plot at {mmd_plot_path}")

def generate_visualizations(func, p_z0, args, result_dir, device, precision, 
                          odeint_func, solver_kwargs, viz_samples, viz_timesteps, t0, t1):
    """Generate flow visualizations."""
    print("Generating visualizations...")
    
    with torch.no_grad():
        # Generate samples flowing forward
        z_t0_sample = p_z0.sample([viz_samples]).to(device)
        logp_diff_t0_sample = torch.zeros(viz_samples, 1, device=device, dtype=torch.float32)
        ts_samples = torch.linspace(t0, t1, viz_timesteps).to(device)
        z_t_samples, logp_diff_samples = odeint_func(
            func,
            (z_t0_sample, logp_diff_t0_sample),
            ts_samples,
            atol=1e-5,
            rtol=1e-5,
            method=args.method,
            **solver_kwargs
        )

        # Generate density on grid
        x_lin = np.linspace(-4, 4, 100)
        y_lin = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_lin, y_lin)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
        logp_diff_grid = torch.zeros(grid_tensor.shape[0], 1, device=device, dtype=torch.float32)
        ts_density = torch.linspace(t1, t0, viz_timesteps).to(device)
        
        z_t_density, logp_diff_density = odeint_func(
            func,
            (grid_tensor, logp_diff_grid),
            ts_density,
            atol=1e-5,
            rtol=1e-5,
            method='rk4', 
            **solver_kwargs
        )

        # Create visualization frames
        for (t, z_sample, z_density, logp_diff) in zip(
                np.linspace(t0, t1, viz_timesteps),
                z_t_samples, z_t_density, logp_diff_density
        ):
            fig = plt.figure(figsize=(12, 4), dpi=200)
            plt.tight_layout()
            plt.axis('off')
            plt.margins(0, 0)
            fig.suptitle(f'{t:.2f}s')
            
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('Target')
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])
            
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('Samples')
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])
            
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('Log Probability')
            ax3.get_xaxis().set_ticks([])
            ax3.get_yaxis().set_ticks([])

            target_sample, _ = get_batch(viz_samples, args.data, device)
            ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                    range=[[-4, 4], [-4, 4]])
            ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                    range=[[-4, 4], [-4, 4]])
            logp_model = p_z0.log_prob(z_density) - logp_diff.view(-1)
            logp_np = np.exp(logp_model.detach().cpu().numpy())
            ax3.tricontourf(grid_points[:, 0], grid_points[:, 1], logp_np, levels=200)
            plt.savefig(os.path.join(result_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                    pad_inches=0.2, bbox_inches='tight')
            plt.close()

        # Create GIF
        imgs = sorted(glob.glob(os.path.join(result_dir, f"cnf-viz-*.jpg")))
        if len(imgs) > 0:
            img, *rest_imgs = [Image.open(f) for f in imgs]
            img.save(fp=os.path.join(result_dir, "cnf-viz.gif"), format='GIF', append_images=rest_imgs,
                    save_all=True, duration=250, loop=0)
        print(f'Saved visualizations for {args.odeint}')


if __name__ == '__main__':
    main()


