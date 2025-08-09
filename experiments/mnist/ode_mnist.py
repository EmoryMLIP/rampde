import os, sys
job_id = os.environ.get("SLURM_JOB_ID", "")
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
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
    parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
    parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
    parser.add_argument('--nepochs', type=int, default=160)
    parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    # new arguments
    parser.add_argument('--method', type=str, choices=['rk4', 'dopri5'], default='rk4')
    parser.add_argument('--precision', type=str, choices=['tfloat32', 'float32', 'float16','bfloat16'], default='float16')
    parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
    parser.add_argument('--no_grad_scaler', action='store_true',
                        help='Disable GradScaler for torchdiffeq with float16 (default: enabled)')
    parser.add_argument('--no_dynamic_scaler', action='store_true',
                        help='Disable DynamicScaler for torchmpnode with float16 (default: enabled)')

    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_freq', type=int, default=50,
                        help='evaluate / log every N training steps')
    return parser


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in , dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        # tt = torch.ones_like(x[:, :1, :, :]) * t
        # ttx = torch.cat([tt, x], 1)
        return self._layer(x)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, method, tol, odeint, loss_scaler=None):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.tol = tol
        self.odeint = odeint
        self.loss_scaler = loss_scaler
        self.integration_time = torch.tensor(np.linspace(0, 1, 32))

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        if self.loss_scaler is not None:
            out = self.odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method, loss_scaler=self.loss_scaler)
        else:
            out = self.odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol, method=self.method)
        return out[-1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)



def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


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
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def main():
    # Create parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set derived boolean values based on flags
    grad_scaler_enabled = not args.no_grad_scaler
    dynamic_scaler_enabled = not args.no_dynamic_scaler
    
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
    
    # Handle adjoint method for torchdiffeq
    if args.odeint == 'torchdiffeq' and args.adjoint:
        try:
            from torchdiffeq import odeint_adjoint as odeint_func
            print("Warning: Using torchdiffeq with adjoint method, which is not recommended for low precision training.")
        except ImportError:
            print("torchdiffeq not available, continuing with torchmpnode")
    
    # Import utilities after setting up the path
    from common import RunningAverageMeter, RunningMaximumMeter, AverageMeter, count_parameters
    
    # Get precision settings
    precision = get_precision_dtype(args.precision)
    
    # Determine scaler configuration
    loss_scaler, scaler_name, loss_scaler_for_odeint = determine_scaler(
        args.odeint, args.precision, grad_scaler_enabled, 
        dynamic_scaler_enabled, DynamicScaler
    )
    
    # Setup experiment directories and logging
    extra_params = {
        'lr': args.lr,
        'nepochs': args.nepochs,
        'batch_size': args.batch_size,
        'data_aug': args.data_aug,
        'downsampling_method': args.downsampling_method
    }
    
    result_dir, ckpt_path, folder_name, device, log_file = setup_experiment(
        args.results_dir, "ode_mnist", "mnist", args.precision,
        args.odeint, args.method, args.seed, args.gpu, scaler_name,
        extra_params=extra_params, args=args
    )
    
    # Save original args to CSV as well (for compatibility)
    args_csv_path = os.path.join(result_dir, "original_args.csv")
    args_dict = vars(args)
    args_df = pd.DataFrame([args_dict])
    args_df.to_csv(args_csv_path, index=False)
    
    # Copy the script to results directory
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))
    
    try:
        is_odenet = args.network == 'odenet'

        if args.downsampling_method == 'conv':
            downsampling_layers = [
                nn.Conv2d(1, 64, 3, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
            ]
        else:  # res
            downsampling_layers = [
                nn.Conv2d(1, 64, 3, 1),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ]

        feature_layers = [ODEBlock(ODEfunc(64), args.method, args.tol, odeint_func, loss_scaler_for_odeint)] if is_odenet \
                         else [ResBlock(64, 64) for _ in range(6)]
        fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

        model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

        print(model)
        print('Number of parameters: {}'.format(count_parameters(model)))

        criterion = nn.CrossEntropyLoss().to(device)

        train_loader, test_loader, train_eval_loader = get_mnist_loaders(
            args.data_aug, args.batch_size, args.test_batch_size
        )
        
        data_gen = inf_generator(train_loader)
        batches_per_epoch = len(train_loader)
        
        lr_fn = learning_rate_with_decay(
            args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch,
            boundary_epochs=[60, 100, 140],
            decay_rates=[1, 0.1, 0.01, 0.001], lr=args.lr
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        best_acc = 0
        fwd_time_meter = AverageMeter()
        bwd_time_meter = AverageMeter()
        mem_meter = RunningMaximumMeter()

        end = time.time()


        csv_path = os.path.join(result_dir, folder_name + ".csv")
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow([
            'iter', 'epoch', 'lr',
            'train_acc', 'val_acc',
            'time_fwd', 'time_bwd', 'time_fwd_sum', 'time_bwd_sum',
            'max_memory_mb'
        ])

        for itr in range(args.nepochs * batches_per_epoch):

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_fn(itr)

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
                loss = criterion(logits, y)
            
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
                optimizer.step()

            torch.cuda.synchronize()
            bwd_time = time.perf_counter() - bwd_start

            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

            fwd_time_meter.update(fwd_time)
            bwd_time_meter.update(bwd_time)
            mem_meter.update(peak_memory)
            
            end = time.time()

            # Check for NaN or infinite loss
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
                break
            
            # Check for NaN gradients (only if not using gradient scaling)
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
                    break

            # evaluate / log every test_freq steps
            if itr % args.test_freq == 0:
                epoch = itr // batches_per_epoch
                with torch.no_grad():
                    train_acc = accuracy(model, train_eval_loader, device)
                    test_acc = accuracy(model, test_loader, device)
                    if test_acc > best_acc:
                        torch.save(
                            {'state_dict': model.state_dict(), 'args': args}, ckpt_path)
                        best_acc = test_acc

                    print(
                        "Iter {:06d} | Epoch {:04d} | LR {:.4f} | "
                        "Fwd {:.3f}s | Bwd {:.3f}s | "
                        "Train Acc {:.4f} | Val Acc {:.4f} | Max Mem {:.1f}MB".format(
                            itr, epoch, optimizer.param_groups[0]['lr'],
                            fwd_time_meter.avg, bwd_time_meter.avg,
                            train_acc, test_acc, mem_meter.max
                        )
                    )

                # write metrics row
                writer.writerow([
                    itr,
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    train_acc,
                    test_acc,
                    fwd_time_meter.avg,
                    bwd_time_meter.avg,
                    fwd_time_meter.sum,
                    bwd_time_meter.sum,
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
            import sys
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()
