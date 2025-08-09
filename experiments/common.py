"""
Common utilities for torchmpnode experiments.

This module provides shared functions for setting up experiments, handling
precision configurations, and managing gradient scaling strategies.
"""

import os
import sys
import shutil
import datetime
import torch
import pandas as pd
from typing import Tuple, Optional, Union


def setup_environment(odeint_type: str, base_dir: str) -> Tuple:
    """
    Setup the environment and imports based on ODE solver type.
    
    Args:
        odeint_type: 'torchmpnode' or 'torchdiffeq'
        base_dir: Base directory of the torchmpnode project
        
    Returns:
        Tuple of (odeint_func, DynamicScaler_class or None)
    """
    # Set up paths for imports
    sys.path.insert(0, os.path.join(base_dir, "examples"))  # for datasets
    
    if odeint_type == 'torchmpnode':
        print("Using torchmpnode")
        from torchmpnode import odeint
        from torchmpnode.loss_scalers import DynamicScaler
        return odeint, DynamicScaler
    else:    
        print("Using torchdiffeq")
        from torchdiffeq import odeint
        return odeint, None


def get_precision_dtype(precision_str: str) -> torch.dtype:
    """
    Convert precision string to torch dtype.
    
    Args:
        precision_str: One of 'float32', 'tfloat32', 'float16', 'bfloat16'
        
    Returns:
        Corresponding torch.dtype
        
    Raises:
        ValueError: If precision_str is not recognized
    """
    precision_map = {
        'float32': torch.float32,
        'tfloat32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    if precision_str not in precision_map:
        raise ValueError(f"Unknown precision: {precision_str}. Must be one of {list(precision_map.keys())}")
    
    return precision_map[precision_str]


def setup_precision(precision_str: str) -> None:
    """
    Setup precision-related backend settings.
    
    Args:
        precision_str: One of 'float32', 'tfloat32', 'float16', 'bfloat16'
    """
    if precision_str == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("Using strict float32 precision")
    elif precision_str == 'tfloat32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Using TF32 precision")
    # float16 and bfloat16 don't need special backend settings


def determine_scaler(
    odeint_type: str, 
    precision_str: str, 
    grad_scaler_enabled: bool, 
    dynamic_scaler_enabled: bool, 
    DynamicScaler_class: Optional[type]
) -> Tuple[Optional[object], Optional[str], Optional[object]]:
    """
    Determine which scaler to use based on configuration.
    
    Args:
        odeint_type: 'torchmpnode' or 'torchdiffeq'
        precision_str: Precision mode string
        grad_scaler_enabled: Whether PyTorch GradScaler is enabled
        dynamic_scaler_enabled: Whether torchmpnode DynamicScaler is enabled
        DynamicScaler_class: The DynamicScaler class (None for torchdiffeq)
        
    Returns:
        Tuple of (scaler_instance, scaler_name, loss_scaler_for_odeint)
        - scaler_instance: GradScaler or None (used for backward pass)
        - scaler_name: 'grad', 'dynamic', 'none', or None
        - loss_scaler_for_odeint: DynamicScaler instance or False or None
        
    The three main cases for torchmpnode + float16:
    1. Dynamic scaling only: dynamic_scaler=True, grad_scaler=False
       → (None, 'dynamic', DynamicScaler instance)
    2. Gradient scaling only: dynamic_scaler=False, grad_scaler=True  
       → (GradScaler, 'grad', False) - False disables internal scaling
    3. No scaling (unsafe): dynamic_scaler=False, grad_scaler=False
       → (None, 'none', None) - None means no scaling at all
    """
    precision = get_precision_dtype(precision_str)
    
    # Only need scaling for float16
    if precision_str != 'float16':
        return None, None, None
    
    # Handle torchdiffeq case
    if odeint_type == 'torchdiffeq':
        if grad_scaler_enabled:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
            print(f"Using PyTorch GradScaler for float16 precision with torchdiffeq (initial scale: {scaler.get_scale()})")
            return scaler, 'grad', None
        else:
            print("WARNING: Using float16 with torchdiffeq without GradScaler - may encounter NaN/overflow")
            return None, 'none', None
    
    # Handle torchmpnode cases
    if odeint_type == 'torchmpnode':
        if dynamic_scaler_enabled and not grad_scaler_enabled:
            # Case 1: Dynamic scaling only
            if DynamicScaler_class is None:
                raise ValueError("DynamicScaler class not available but dynamic scaling is enabled")
            scaler = DynamicScaler_class(precision)
            print("Using DynamicScaler for float16 precision with torchmpnode")
            return None, 'dynamic', scaler
            
        elif not dynamic_scaler_enabled and grad_scaler_enabled:
            # Case 2: Gradient scaling only (safe mode)
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
            print(f"Using PyTorch GradScaler for float16 precision with torchmpnode (safe mode, DynamicScaler disabled)")
            print(f"Initial scale: {scaler.get_scale()}")
            return scaler, 'grad', False  # False disables internal scaling
            
        elif not dynamic_scaler_enabled and not grad_scaler_enabled:
            # Case 3: No scaling (unsafe mode)
            print("WARNING: Using float16 with torchmpnode without any scaling (unsafe mode)")
            print("Training will stop if NaN/inf is detected in loss or gradients")
            return None, 'none', False
            
        else:
            # Invalid case: both scalers enabled
            raise ValueError("Cannot enable both GradScaler and DynamicScaler simultaneously")
    
    raise ValueError(f"Unknown odeint type: {odeint_type}")


def setup_experiment(
    results_dir: str,
    experiment_name: str,
    data_name: str,
    precision_str: str,
    odeint_type: str,
    method: str,
    seed: Optional[int],
    gpu: int,
    scaler_name: Optional[str],
    timestamp: Optional[str] = None,
    extra_params: Optional[dict] = None,
    args: Optional[object] = None
) -> Tuple[str, str, str, torch.device, object]:
    """
    Setup experiment directories, logging, and environment.
    
    Args:
        results_dir: Base results directory
        experiment_name: Name of the experiment (e.g., 'otflowlarge', 'stl10')
        data_name: Dataset name (e.g., 'bsds300', 'miniboone')
        precision_str: Precision mode string
        odeint_type: 'torchmpnode' or 'torchdiffeq'
        method: ODE solver method (e.g., 'rk4', 'euler')
        seed: Random seed (None for no seed)
        gpu: GPU device number
        scaler_name: Name of the scaler ('grad', 'dynamic', 'none', or None)
        timestamp: Optional timestamp string (auto-generated if None)
        extra_params: Optional dict of extra parameters to include in folder name
        args: Optional argument parser namespace containing all experiment parameters
        
    Returns:
        Tuple of (result_dir, ckpt_path, folder_name, device, log_file)
    """
    job_id = os.environ.get("SLURM_JOB_ID", "")
    
    os.makedirs(results_dir, exist_ok=True)
    seed_str = f"seed{seed}" if seed is not None else "noseed"
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build folder name with scaler type
    precision_scaler = precision_str
    if scaler_name:
        precision_scaler = f"{precision_str}_{scaler_name}"
    
    # Build basic folder name components
    folder_parts = [data_name, precision_scaler, odeint_type, method]
    
    # Add extra parameters if provided
    if extra_params:
        for key, value in extra_params.items():
            folder_parts.append(f"{key}_{value}")
    
    folder_parts.extend([seed_str, timestamp])
    folder_name = "_".join(folder_parts)
    
    result_dir = os.path.join(results_dir, experiment_name, folder_name)
    ckpt_path = os.path.join(result_dir, 'ckpt.pth')
    os.makedirs(result_dir, exist_ok=True)
    
    # Save result directory path for reference
    with open("result_dir.txt", "w") as f:
        f.write(result_dir)
    
    # Save all parameters to CSV for easy loading
    args_dict = {
        'experiment_name': experiment_name,
        'data_name': data_name,
        'precision_str': precision_str,
        'odeint_type': odeint_type,
        'method': method,
        'seed': seed,
        'gpu': gpu,
        'scaler_name': scaler_name,
        'timestamp': timestamp,
        'job_id': job_id
    }
    
    # Include all arguments from the parser if provided
    if args is not None:
        args_dict.update(vars(args))
    
    # Add any extra parameters (this allows overriding args if needed)
    if extra_params:
        args_dict.update(extra_params)
    
    args_csv_path = os.path.join(result_dir, "args.csv")
    args_df = pd.DataFrame([args_dict])
    args_df.to_csv(args_csv_path, index=False)
    
    # Redirect stdout and stderr to a log file
    log_path = os.path.join(result_dir, folder_name + ".txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    
    device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
    
    # Setup precision (already done but included for completeness)
    setup_precision(precision_str)
    
    torch.backends.cudnn.benchmark = True
    
    # Print environment and hardware info
    print("Environment Info:")
    print(f"  Python version: {sys.version}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    try:
        print(f"  CUDA version: {torch.version.cuda}")
    except:
        print(f"  CUDA version: N/A")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"  GPU Device Name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")
    print(f"  Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    
    print(f"\nExperiment started at {datetime.datetime.now()}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {data_name}")
    print(f"Precision: {precision_str} (scaler: {scaler_name or 'none'})")
    print(f"ODE solver: {odeint_type} with method {method}")
    print(f"Seed: {seed}")
    print(f"Results will be saved in: {result_dir}")
    print(f"SLURM job id: {job_id}")
    print(f"Model checkpoint path: {ckpt_path}")
    
    if extra_params:
        print("\nExtra parameters:")
        for key, value in extra_params.items():
            print(f"  {key}: {value}")
    
    return result_dir, ckpt_path, folder_name, device, log_file


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class RunningMaximumMeter(object):
    """Computes and stores the maximum value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.max = float('-inf')

    def update(self, val):
        if self.val is None:
            self.max = val
        else:
            self.max = max(self.max, val)
        self.val = val


class AverageMeter(object):
    """Computes and stores the cumulative average, sum, count, and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)