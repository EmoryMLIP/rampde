import os, sys
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.amp import autocast

from ode_mnist import conv3x3, conv1x1, norm, ResBlock, ConcatConv2d, ODEfunc, ODEBlock, Flatten, get_mnist_loaders, one_hot, accuracy, count_parameters, inf_generator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint


if __name__ == '__main__':
    # Settings
    gpu = 0
    data_aug = True
    batch_size = 128
    test_batch_size = 1000
    nepochs = 1  # we'll just do one iteration test
    lr = 0.1
    network = 'odenet'  # ODE network
    tol = 1e-3
    adjoint = False
    odeint_choice = 'torchdiffeq'  # Using torchdiffeq

    
    device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Import odeint based on choice
    # using torchdiffeq
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # Methods and precisions to test
    methods = ['rk4', 'dopri5']
    precisions = ['float32', 'float16', 'bfloat16']
    odeints = [odeint, mpodeint]


    precision_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }

    # Prepare dataset just once
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(data_aug, batch_size, test_batch_size)
    data_gen = inf_generator(train_loader)
    # Get a batch
    x, y = next(data_gen)
    x = x.to(device)
    y = y.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)


    # Build model
    downsampling_layers = [
        nn.Conv2d(1, 64, 3, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
    ]
            
    results = []
    for odeint in odeints:
        for method in methods:
            if odeint == mpodeint and method == 'dopri5':
                    # Skip this combination, as it is not supported
                    continue
            
            feature_layers = [ODEBlock(ODEfunc(64), method=method, tol=tol, odeint=odeint)]
            fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
            model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
        
            for prec_str in precisions:
                chosen_precision = precision_map[prec_str]

                if odeint != mpodeint and (prec_str == 'bfloat16' or prec_str == 'float16'):
                    # Skip this combination, as it is not supported
                    continue


                
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                start_time = time.perf_counter()

                with autocast(device_type='cuda', dtype=chosen_precision):
                    logits = model(x)
                    loss = criterion(logits, y)

                # Backward and optimize
                loss.backward()
                
                torch.cuda.synchronize(device)
                end_time = time.perf_counter()

                elapsed_time = end_time - start_time
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

                results.append((method, prec_str, elapsed_time, peak_memory, logits.dtype, loss.dtype))

        # Print results as a table
        print(f"{'Method':<10} {'Precision':<10} {'Time (s)':<10} {'Peak Mem (MB)':<15} {'logits dtype':<12} {'loss dtype'}")
        print("-" * 50)
        for (method, prec_str, elapsed_time, peak_memory, dtype_logits, dtype_loss) in results:
            print(f"{method:<10} {prec_str:<10} {elapsed_time:<10.4f} {peak_memory:<15.2f} {str(dtype_logits):<12} {str(dtype_loss)}")