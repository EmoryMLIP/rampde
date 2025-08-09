import os, sys
job_id = os.environ.get("SLURM_JOB_ID", "")
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
# import torchvision.datasets as datasets
from torchvision.datasets import STL10
import torchvision.transforms as transforms
from torch.amp import autocast
import time
import datetime

import csv
import shutil
import sys

import pandas as pd
import matplotlib.pyplot as plt

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
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--nepochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # new arguments
    parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
    parser.add_argument('--precision', type=str, choices=['tfloat32', 'float32', 'float16','bfloat16'], default='float16')
    parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
    parser.add_argument('--unstable', action='store_true', 
                        help='Use unstable ODE formulation (default: stable)')
    parser.add_argument('--no_grad_scaler', action='store_true',
                        help='Disable GradScaler for torchdiffeq with float16 (default: enabled)')
    parser.add_argument('--no_dynamic_scaler', action='store_true',
                        help='Disable DynamicScaler for torchmpnode with float16 (default: enabled)')

    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_freq', type=int, default=1,
                        help='evaluate / log every N training steps')
    parser.add_argument('--width', type=int, default=64,
                        help='Base channel width (default: 64)')
    return parser



class ODEFunc(nn.Module):
    """
    Time-dependent ODE function with piecewise constant weights.
    
    This implementation uses piecewise constant weights that change at discrete time intervals.
    The weights are held constant within each interval and use left-neighbor interpolation.
    This approach is compatible with any ODE solver (Euler, RK4, adaptive methods, etc.)
    since it doesn't rely on specific intermediate time points.
    
    For a time grid [t0, t1, t2, ..., tn], we create n intervals:
    - [t0, t1): uses weights[0]
    - [t1, t2): uses weights[1] 
    - ...
    - [tn-1, tn]: uses weights[n-1]
    
    * One Conv2d (`self.A`) is created.
    * A learnable weight bank (n_intervals×C×C×3×3) and bias bank (n_intervals×C) are stored.
    * At each call we reassign buffers to point to the current interval weights (no copy).
    """
    def __init__(self, ch, t_grid, act=nn.ReLU(inplace=True),is_stable=True):
        super().__init__()

        n_steps = len(t_grid)-1
        
        # Create a weight bank
        init_weight = torch.randn(ch, ch, 3, 3).mul_(0.1)
        init_bias = torch.zeros(ch)
        
        # Use the exact same pattern as ODEFunc for consistency
        # Create weight banks for piecewise constant weights (one per interval)
        self.weight_bank = nn.Parameter(init_weight.unsqueeze(0).repeat(n_steps, 1, 1, 1, 1))
        self.bias_bank = nn.Parameter(init_bias.unsqueeze(0).repeat(n_steps, 1))
        
        # Create a single conv and register buffers for its weights
        self.A = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.A._parameters.pop('weight')
        self.A._parameters.pop('bias')
        self.A.register_buffer('weight', self.weight_bank[0])
        self.A.register_buffer('bias', self.bias_bank[0])
        
        # For transpose convolution
        self.A_T = nn.ConvTranspose2d(ch, ch, 3, padding=1, bias=False)
        self.A_T._parameters.pop('weight')
        self.A_T.register_buffer('weight', self.weight_bank[0])

        # per-step norms for each interval (one per original time step)
        self.norms = nn.ModuleList([nn.InstanceNorm2d(ch, affine=False)
                                     for _ in range(n_steps)])

        # Store original time grid parameters for piecewise constant interpolation
        self.t_start = float(t_grid[0])
        self.t_end = float(t_grid[-1])
        self.n_intervals = n_steps  # number of intervals between time points
        
        self.act = act
        self.is_stable = is_stable
        if is_stable:
            self.factor = -1.0
        else:
            self.factor = 1.0


    def forward(self, t, y):
        """
        Piecewise constant weights with left-neighbor interpolation.
        
        For any time t in [t_start, t_end], find the interval it belongs to:
        - Normalize t to [0, 1] range
        - Scale by number of intervals and take floor for left-neighbor
        - Clamp to valid range to handle boundary cases
        
        This works with any ODE solver since it doesn't depend on specific
        intermediate evaluation points.
        """
        # Normalize t to [0, 1] range of the grid  
        t_normalized = (t.item() - self.t_start) / (self.t_end - self.t_start)
        
        # Scale to interval index and take floor for left-neighbor
        # Clamp to ensure we stay within valid indices [0, n_intervals-1]
        idx = max(0, min(int(t_normalized * self.n_intervals), self.n_intervals - 1))

        # Set the current weights and biases for this interval
        self.A._buffers['weight'] = self.weight_bank[idx]
        self.A._buffers['bias'] = self.bias_bank[idx]
        self.A_T._buffers['weight'] = self.weight_bank[idx]

        y = self.A(y)
        y = self.act(y)
        y = self.norms[idx](y)
        y = self.A_T(y)

        return self.factor*y
    
class ODEBlock(nn.Module):
    def __init__(self, func, t_grid, solver="rk4", steps=4, loss_scaler=None, odeint_func=None):
        super().__init__()
        self.func   = func
        self.solver = solver
        # Register t_grid as a buffer so it moves with the model
        self.register_buffer('t_grid', t_grid)
        self.loss_scaler = loss_scaler
        self.odeint_func = odeint_func

    def forward(self, x):
        if self.loss_scaler is not None:
            out = self.odeint_func(self.func, x, self.t_grid, method=self.solver, loss_scaler=self.loss_scaler)
        else:
            out = self.odeint_func(self.func, x, self.t_grid, method=self.solver)
        return out[-1]

class MPNODE_STL10(nn.Module):
    def __init__(self, width, args, precision, odeint_func, ScalerClass, dynamic_scaler_enabled=False, grad_scaler_enabled=False):
        super().__init__()
        ch = width
        # Create t_grid on the appropriate device
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        t_grid = torch.linspace(0, 1.0, 5, device=device)
        # 1) stem: 3×96×96 -> 64×96×96
        self.stem = nn.Conv2d(3, ch, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)

        # 2) ODE block #1
        if args.odeint == 'torchmpnode' and dynamic_scaler_enabled and ScalerClass is not None:
            S1 = ScalerClass(precision)
        elif args.odeint == 'torchmpnode' and args.precision == 'float16' and not dynamic_scaler_enabled:
            # Explicitly disable internal scaler when using external GradScaler  
            S1 = False
        else:
            S1 = None
        self.ode1 = ODEBlock(ODEFunc(ch, t_grid, is_stable=args.stable), t_grid, solver="rk4", steps=4, loss_scaler=S1, odeint_func=odeint_func)
        
        # 3) down-sample stride-2 3×3
        self.conn1 = nn.Conv2d(ch, 2*ch, 1,  padding=0, bias=True)
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.norm3 = nn.InstanceNorm2d(2*ch, affine=True)
        # self.norm3 = nn.InstanceNorm2d(ch)

        # 4) ODE block #2
        if args.odeint == 'torchmpnode' and dynamic_scaler_enabled and ScalerClass is not None:
            S2 = ScalerClass(precision)
        elif args.odeint == 'torchmpnode' and args.precision == 'float16' and not dynamic_scaler_enabled:
            # Explicitly disable internal scaler when using external GradScaler
            S2 = False
        else:
            S2 = None
        self.ode2 = ODEBlock(ODEFunc(2*ch, t_grid, is_stable=args.stable), t_grid, solver="rk4", steps=4, loss_scaler=S2, odeint_func=odeint_func)
        self.conn2 = nn.Conv2d(2*ch, 4*ch, 1,  padding=0, bias=True)
        self.avg2 = nn.AvgPool2d(2, stride=2)
        self.norm4 = nn.InstanceNorm2d(4*ch, affine=True)
        
        if args.odeint == 'torchmpnode' and dynamic_scaler_enabled and ScalerClass is not None:
            S3 = ScalerClass(precision)
        elif args.odeint == 'torchmpnode' and args.precision == 'float16' and not dynamic_scaler_enabled:
            # Explicitly disable internal scaler when using external GradScaler
            S3 = False
        else:
            S3 = None
        self.ode3 = ODEBlock(ODEFunc(4*ch, t_grid, is_stable=args.stable), t_grid, solver="rk4", steps=4, loss_scaler=S3, odeint_func=odeint_func)
        
        self.act = nn.ReLU(inplace=True)
        # 5) global avg-pool + FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # 128×1×1
            nn.Flatten(),
            nn.Linear(4*ch, 10)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.ode1(x)
        # x = self.norm2(x)
        x = self.conn1(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.avg1(x)
        
        x = self.ode2(x)
        x = self.conn2(x)
        x = self.norm4(x)
        x = self.act(x)
        x = self.avg2(x)
        
        x = self.ode3(x)
        
        return self.head(x)


def worker_init_fn(worker_id):
    """Initialize worker with proper seeding for reproducibility."""
    # Get the base random seed from the main process
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Note: torch seed is automatically handled for DataLoader workers

def get_stl10_loaders(batch_size=128,
                      test_batch_size=1000,
                      perc=1.0,
                      seed=None):
    """Return train_loader, test_loader, train_eval_loader for STL-10.

    Parameters
    ----------
    data_aug : bool          – if True, use random crop + flip.
    batch_size : int
    test_batch_size : int
    perc : float             – unused (kept for interface compatibility).

    All loaders use drop_last=True so the batch count is deterministic.
    """

    # normalization constants for STL-10 RGB
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2241, 0.2210, 0.2239)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # optional regularizers:
        # transforms.RandomErasing(p=0.2, scale=(0.02,0.33), ratio=(0.3,3.3)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # ----- full 5k train set (load twice with different transforms) -----
    full_train_aug  = STL10(root='.data/stl10', split='train',
                            download=True, transform=transform_train)
    full_train_eval = STL10(root='.data/stl10', split='train',
                            download=True, transform=transform_test)

    # ----- deterministic split -----
    # Use provided seed for data split, or default to 42 for backward compatibility
    split_seed = seed if seed is not None else 42
    g = torch.Generator().manual_seed(split_seed)
    idx = torch.randperm(len(full_train_aug), generator=g)
    idx_train, idx_val = idx[:int(4000*perc)], idx[4000:]          # 4 k / 1 k

    train_set = Subset(full_train_aug,  idx_train.tolist())
    val_set   = Subset(full_train_eval, idx_val.tolist())         # no augmentation
    train_eval_set   = Subset(full_train_eval, idx_train.tolist())         # no augmentation

    # ----- loaders with proper seeding -----
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, drop_last=True,
                              worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_set,   batch_size=test_batch_size,
                              shuffle=False, num_workers=2,
                              worker_init_fn=worker_init_fn)
    train_eval_loader   = DataLoader(train_eval_set,   batch_size=test_batch_size,
                              shuffle=False, num_workers=2,
                              worker_init_fn=worker_init_fn)
    return train_loader, val_loader, train_eval_loader



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates, lr):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, device):
    loss = 0.0
    total_correct = 0
    N = len(dataset_loader.dataset)
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y).item() * y.size(0)
        predicted_class = logits.argmax(dim=1)
        total_correct += (predicted_class == y).sum().item()
    return total_correct / N, loss/N




def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



def main():
    # Create parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set derived boolean values based on flags
    args.stable = not args.unstable  # stable is True unless --unstable is passed
    grad_scaler_enabled = not args.no_grad_scaler  # grad_scaler is True unless --no_grad_scaler is passed
    dynamic_scaler_enabled = not args.no_dynamic_scaler  # dynamic_scaler is True unless --no_dynamic_scaler is passed
    
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
    
    # Get base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Setup environment and imports
    odeint_func, DynamicScaler = setup_environment(args.odeint, base_dir)
    
    # Import utilities after setting up the path
    from common import RunningAverageMeter, RunningMaximumMeter, AverageMeter, count_parameters
    
    # Get precision settings
    precision = get_precision_dtype(args.precision)
    
    # Determine scaler configuration (need to get all 3 return values)
    loss_scaler, scaler_name, loss_scaler_for_odeint = determine_scaler(
        args.odeint, args.precision, grad_scaler_enabled, 
        dynamic_scaler_enabled, DynamicScaler
    )
    
    # Setup experiment directories and logging
    stable_str = "stable" if args.stable else "unstable"
    extra_params = {
        'stable': stable_str,
        'lr': args.lr,
        'nepochs': args.nepochs,
        'batch_size': args.batch_size,
        'width': args.width
    }
    
    result_dir, ckpt_path, folder_name, device, log_file = setup_experiment(
        args.results_dir, "ode_stl10", "stl10", args.precision,
        args.odeint, args.method, args.seed, args.gpu, scaler_name,
        extra_params=extra_params, args=args
    )
    
    # Copy the script to results directory
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))
    
    try:
        # Create model
        model = MPNODE_STL10(args.width, args, precision, odeint_func, DynamicScaler, 
                           dynamic_scaler_enabled, grad_scaler_enabled).to(device)
        print(model)
        print('Number of parameters: {}'.format(count_parameters(model)))

        criterion = nn.CrossEntropyLoss().to(device)

        train_loader, test_loader, train_eval_loader = get_stl10_loaders(
             args.batch_size, args.test_batch_size, seed=args.seed
        )

        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs*batches_per_epoch, eta_min=1e-4)
        # scheduler that does nothing

        best_acc = 0
        train_loss_meter = RunningAverageMeter()
        fwd_time_meter = AverageMeter()
        bwd_time_meter = AverageMeter()
        mem_meter = RunningMaximumMeter()

        csv_path = os.path.join(result_dir, folder_name + ".csv")
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow([
            'iter', 'epoch', 'lr',
            'running_loss', 'train_loss', 'val_loss',
            'time_fwd', 'time_bwd', 'time_fwd_sum', 'time_bwd_sum',
            'train_acc', 'val_acc', 'max_memory_mb'
        ])

        for itr in range(args.nepochs * batches_per_epoch):
            
            optimizer.zero_grad()
            x, y = data_gen.__next__()
            x = x.to(device)
            y = y.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            
            # Time forward pass
            torch.cuda.synchronize()
            fwd_start = time.perf_counter()
            
            with autocast(device_type='cuda', dtype=precision):
                logits = model(x)
                loss = criterion(logits.float(), y)
            
            torch.cuda.synchronize()
            fwd_time = time.perf_counter() - fwd_start
                
            # Time backward pass
            torch.cuda.synchronize()
            bwd_start = time.perf_counter()
            
            # Handle backward pass with or without loss scaling
            if loss_scaler is not None:
                # Track loss scale before step
                old_scale = loss_scaler.get_scale()
                
                # Use gradient scaling for torchdiffeq with float16
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
                
                # Track loss scale after step and log changes
                new_scale = loss_scaler.get_scale()
                if old_scale != new_scale:
                    print(f"Iteration {itr}: Loss scale changed from {old_scale} to {new_scale} (gradient overflow detected)")
                elif itr < 20 or itr % 100 == 0:  # Log scale periodically for first 20 iterations or every 100
                    print(f"Iteration {itr}: Loss scale = {new_scale} (no overflow)")
                
                # Only step scheduler if no overflow occurred (scale didn't change)
                if old_scale == new_scale:
                    scheduler.step()
                else:
                    print(f"Iteration {itr}: Skipping scheduler step due to gradient overflow")
            else:
                # Standard backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            for param in model.parameters():
                param.data = param.data.clamp_(-1, 1)
            
            torch.cuda.synchronize()
            bwd_time = time.perf_counter() - bwd_start
            
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

            # Check for NaN or infinite loss (outside timed zone)
            if not torch.isfinite(loss).all():
                print(f"Training stopped at iteration {itr}: Loss is {'NaN' if torch.isnan(loss).any() else 'infinite'}")
                print(f"Loss value: {loss.item()}")
                print("Saving current model state before stopping...")
                torch.save({
                    'state_dict': model.state_dict(), 
                    'args': args,
                    'iteration': itr,
                    'loss': loss.item()
                }, ckpt_path.replace('.pth', '_emergency_stop.pth'))
                return  # Exit the training function
            
            # Check for NaN gradients (outside timed zone)
            # Only stop training for NaN gradients if we're not using gradient scaling
            # When using GradScaler, NaN/inf gradients are expected and handled automatically
            if loss_scaler is None:
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"Training stopped at iteration {itr}: NaN/infinite gradient detected in parameter '{name}'")
                        print(f"Gradient stats - min: {param.grad.min().item()}, max: {param.grad.max().item()}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print("Saving current model state before stopping...")
                    torch.save({
                        'state_dict': model.state_dict(), 
                        'args': args,
                        'iteration': itr,
                        'loss': loss.item()
                    }, ckpt_path.replace('.pth', '_gradient_nan_stop.pth'))
                    return  # Exit the training function
            else:
                # When using gradient scaling, just log if we encounter NaN gradients
                # but don't stop training as GradScaler handles this automatically
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        print(f"NaN/inf gradients detected in '{name}' at iteration {itr} - GradScaler will handle this")
                        has_nan_grad = True
                        break
                
                # Also log if we have finite gradients for comparison
                if not has_nan_grad and (itr < 5 or itr % 50 == 0):
                    print(f"Iteration {itr}: All gradients are finite")

            fwd_time_meter.update(fwd_time)
            bwd_time_meter.update(bwd_time)
            train_loss_meter.update(loss.item())
            mem_meter.update(peak_memory)

            # evaluate / log every test_freq steps
            if itr % batches_per_epoch*args.test_freq == 0:
                epoch = itr // batches_per_epoch

                with torch.no_grad():
                    with autocast(device_type='cuda', dtype=precision):
                        train_acc, train_loss = accuracy(model, train_eval_loader, device)
                        val_acc, val_loss = accuracy(model, test_loader, device)
                        if val_acc > best_acc:
                            torch.save(
                                {'state_dict': model.state_dict(), 'args': args}, ckpt_path)
                            best_acc = val_acc

                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        "Iter {:06d} | Epoch {:04d} | LR {:.4f} | "
                        "Running Loss {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | "
                        "Fwd {:.3f}s | Bwd {:.3f}s | "
                        "Train Acc {:.4f} | Val Acc {:.4f} | Max Mem {:.1f}MB".format(
                            itr, epoch, current_lr,
                            train_loss_meter.val, train_loss, val_loss,
                            fwd_time_meter.avg, bwd_time_meter.avg,
                            train_acc, val_acc, mem_meter.max
                        )
                    )

                # write metrics row
                writer.writerow([
                    itr,
                    epoch,
                    current_lr,
                    train_loss_meter.val,
                    train_loss,
                    val_loss,
                    fwd_time_meter.avg,
                    bwd_time_meter.avg,
                    fwd_time_meter.sum,
                    bwd_time_meter.sum,
                    train_acc,
                    val_acc,
                    mem_meter.max
                ])
                csv_file.flush()


        csv_file.close()


        df = pd.read_csv(csv_path)

        # 1) accuracy plot
        plt.figure(figsize=(6, 4))
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        acc_plot = os.path.join(result_dir, 'accuracy.png')
        plt.savefig(acc_plot, bbox_inches='tight')
        plt.close()
        print(f"Saved accuracy plot at {acc_plot}")

        
    finally:
        # Close log file to restore stdout/stderr
        if 'log_file' in locals() and log_file:
            log_file.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()
