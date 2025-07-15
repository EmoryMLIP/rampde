#!/usr/bin/env python3
"""
Precision-aware evaluation script for OTFlowLarge experiments.
Based on https://github.com/EmoryMLIP/OT-Flow/blob/master/evaluateLargeOTflow.py
"""

import os
import sys

# Fix MKL threading issues before importing numpy/torch
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import argparse
import numpy as np  # Import numpy first to initialize MKL properly
import torch
import time
import datetime
from torch.amp import autocast

# Add path for imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(base_dir, "examples"))
sys.path.insert(0, base_dir)

def create_parser():
    """Create argument parser for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate OTFlowLarge model')
    
    # Model and data arguments
    parser.add_argument('--data', type=str, required=True,
                        choices=['miniboone', 'bsds300', 'power', 'gas', 'hepmass'],
                        help='Dataset to evaluate')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Precision and solver arguments
    parser.add_argument('--precision', type=str, required=True,
                        choices=['tfloat32', 'float32', 'float16', 'bfloat16'],
                        help='Precision used during training')
    parser.add_argument('--odeint', type=str, required=True,
                        choices=['torchdiffeq', 'torchmpnode'],
                        help='ODE solver used during training')
    parser.add_argument('--method', type=str, required=True,
                        choices=['rk4', 'euler'],
                        help='Integration method used during training')
    
    # Scaler arguments
    parser.add_argument('--grad_scaler', action='store_true',
                        help='Use GradScaler (as used during training)')
    parser.add_argument('--dynamic_scaler', action='store_true',
                        help='Use DynamicScaler (as used during training)')
    
    # Evaluation parameters
    parser.add_argument('--nt', type=int, default=18,
                        help='Number of integration time steps for evaluation')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size for evaluation')
    parser.add_argument('--alpha', type=str, default='1.0,2000.0,800.0',
                        help='Alpha hyperparameters as comma-separated values')
    
    # Technical arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as checkpoint dir)')
    
    return parser

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

def setup_environment(args):
    """Setup the environment and imports based on args."""
    if args.odeint == 'torchmpnode':
        print("Using torchmpnode for evaluation")
        from torchmpnode import odeint
        from torchmpnode.loss_scalers import DynamicScaler, NoScaler
        
        # Determine appropriate scaler
        if args.precision == 'float16' and args.dynamic_scaler:
            ScalerClass = DynamicScaler
        else:
            ScalerClass = NoScaler
            
        return odeint, ScalerClass
    else:
        print("Using torchdiffeq for evaluation")
        from torchdiffeq import odeint
        return odeint, None

def load_data(name, datasets_module):
    """Load dataset using the datasets module."""
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
        raise ValueError(f'Unknown dataset: {name}')

def check_mkl_environment():
    """Check and report MKL configuration."""
    print("MKL Environment Check:")
    print(f"  MKL_THREADING_LAYER: {os.environ.get('MKL_THREADING_LAYER', 'not set')}")
    print(f"  MKL_SERVICE_FORCE_INTEL: {os.environ.get('MKL_SERVICE_FORCE_INTEL', 'not set')}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    
    # Check if numpy is using MKL
    try:
        import numpy as np
        config = np.show_config()
        print(f"  NumPy linked with MKL: {'mkl' in str(config).lower()}")
    except:
        print("  NumPy MKL check: failed")

def get_minibatch(X, num_samples):
    """Get a minibatch from the dataset."""
    idx = torch.randint(0, X.size(0), (num_samples,), device=X.device)
    x = X[idx]
    B = x.size(0)
    z = torch.zeros(B, 1, dtype=torch.float32, device=X.device)
    return x, z.clone(), z.clone(), z.clone()

def evaluate_model(args):
    """Main evaluation function."""
    # Check MKL environment first
    check_mkl_environment()
    
    # Set random seed
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup precision
    precision = get_precision_dtype(args.precision)
    setup_precision(args.precision)
    print(f"Evaluation precision: {args.precision} ({precision})")
    
    # Setup environment and imports
    odeint_func, ScalerClass = setup_environment(args)
    
    # Import required modules
    from utils import RunningAverageMeter
    from mmd import mmd
    import datasets
    from Phi import Phi
    
    # Parse alpha hyperparameters
    alpha = [float(a) for a in args.alpha.split(',')]
    print(f"Alpha hyperparameters: {alpha}")
    
    # Load dataset
    print(f"Loading dataset: {args.data}")
    data = load_data(args.data, datasets)
    test_x = torch.from_numpy(data.tst.x).float().to(device)
    d = test_x.size(1)
    print(f"Dataset {args.data}: test={test_x.shape}, dimensions={d}")
    
    # Load model checkpoint
    print(f"Loading model from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    
    # Create model (this needs to match the training setup)
    # Import the OTFlow class from the training script
    sys.path.insert(0, os.path.dirname(args.resume))
    from otflowlarge import OTFlow
    
    # Create model with same architecture as training
    func = OTFlow(in_out_dim=d, hidden_dim=256, alpha=alpha, Phi_class=Phi).to(device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        func.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        func.load_state_dict(checkpoint['state_dict'])
    else:
        func.load_state_dict(checkpoint)
    
    func.eval()
    print("Model loaded and set to evaluation mode")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.resume)
    eval_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Evaluation outputs will be saved to: {eval_dir}")
    
    # Setup loss scaler if needed
    loss_scaler = None
    if args.odeint == 'torchmpnode' and ScalerClass is not None:
        loss_scaler = ScalerClass(precision)
    
    # Create covariance matrix and distribution
    cov = torch.eye(d, device=device) * 0.1
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.zeros(d, device=device),
        covariance_matrix=cov
    )
    
    # Evaluation parameters
    t0, t1 = 0.0, 1.0
    
    print("\n" + "="*50)
    print("STARTING MODEL EVALUATION")
    print("="*50)
    
    # Evaluate on test data
    with torch.no_grad():
        print(f"Evaluating on test data ({test_x.shape[0]} samples)...")
        
        test_loss_meter = RunningAverageMeter()
        test_nll_meter = RunningAverageMeter()
        test_L_meter = RunningAverageMeter()
        test_HJB_meter = RunningAverageMeter()
        
        num_batches = (test_x.size(0) + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, test_x.size(0))
            
            # Get batch
            batch_x = test_x[start_idx:end_idx]
            z0, logp0, cL0, cH0 = get_minibatch(batch_x, batch_x.size(0))
            
            # Forward pass with same precision as training
            with autocast(device_type='cuda', dtype=precision):
                ts = torch.linspace(t0, t1, args.nt, device=device)
                
                if loss_scaler is not None:
                    z_t, logp_t, cL_t, cH_t = odeint_func(
                        func, (z0, logp0, cL0, cH0), ts,
                        method=args.method, loss_scaler=loss_scaler
                    )
                else:
                    z_t, logp_t, cL_t, cH_t = odeint_func(
                        func, (z0, logp0, cL0, cH0), ts,
                        method=args.method
                    )
                
                z1, logp1, cL1, cH1 = z_t[-1], logp_t[-1], cL_t[-1], cH_t[-1]
                logp_x = p_z0.log_prob(z1).view(-1, 1) + logp1
                loss = (-alpha[2]*logp_x.mean() + alpha[0]*cL1.mean() + alpha[1]*cH1.mean())
            
            # Update meters
            test_loss_meter.update(loss.item())
            test_nll_meter.update((-logp_x.mean()).item())
            test_L_meter.update(cL1.mean().item())
            test_HJB_meter.update(cH1.mean().item())
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                print(f"Batch {batch_idx+1}/{num_batches}: "
                      f"Loss={loss.item():.4f}, NLL={(-logp_x.mean()).item():.4f}")
        
        print(f"\nTest Results:")
        print(f"  Average Loss: {test_loss_meter.avg:.6f}")
        print(f"  Average NLL:  {test_nll_meter.avg:.6f}")
        print(f"  Average L:    {test_L_meter.avg:.6f}")
        print(f"  Average HJB:  {test_HJB_meter.avg:.6f}")
    
    # Generate samples for MMD calculation
    print(f"\nGenerating samples for MMD calculation...")
    N = min(args.batch_size, test_x.size(0))
    
    with torch.no_grad():
        # Forward transformation: data -> latent
        test_samples = test_x[:N]
        z0, logp0, cL0, cH0 = get_minibatch(test_samples, N)
        
        with autocast(device_type='cuda', dtype=precision):
            ts = torch.linspace(t0, t1, args.nt, device=device)
            
            if loss_scaler is not None:
                z_t, _, _, _ = odeint_func(
                    func, (z0, logp0, cL0, cH0), ts,
                    method=args.method, loss_scaler=loss_scaler
                )
            else:
                z_t, _, _, _ = odeint_func(
                    func, (z0, logp0, cL0, cH0), ts,
                    method=args.method
                )
        
        z_fwd = z_t[-1]
        model_latent = z_fwd.cpu().numpy()
        
        # Inverse transformation: latent -> data
        y = p_z0.sample([N]).to(device)
        logp0_inv = torch.zeros(N, 1, device=device)
        cL0_inv = torch.zeros_like(logp0_inv)
        cH0_inv = torch.zeros_like(logp0_inv)
        
        with autocast(device_type='cuda', dtype=precision):
            ts_inv = torch.linspace(t1, t0, args.nt, device=device)
            
            if loss_scaler is not None:
                z_inv_t, _, _, _ = odeint_func(
                    func, (y, logp0_inv, cL0_inv, cH0_inv), ts_inv,
                    method=args.method, loss_scaler=loss_scaler
                )
            else:
                z_inv_t, _, _, _ = odeint_func(
                    func, (y, logp0_inv, cL0_inv, cH0_inv), ts_inv,
                    method=args.method
                )
        
        z_inv = z_inv_t[-1]
        model_samples = z_inv.cpu().numpy()
        test_samples_np = test_samples.cpu().numpy()
        
        # Calculate MMD
        mmd_val = mmd(model_samples, test_samples_np)
        print(f"MMD between generated and test samples: {mmd_val:.6e}")
    
    # Save evaluation results
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': vars(args),
        'test_loss': test_loss_meter.avg,
        'test_nll': test_nll_meter.avg,
        'test_L': test_L_meter.avg,
        'test_HJB': test_HJB_meter.avg,
        'mmd': mmd_val,
        'num_test_samples': test_x.size(0),
        'evaluation_samples': N,
    }
    
    import json
    results_file = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_file}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED")
    print("="*50)
    
    return results

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("OTFlowLarge Model Evaluation")
    print(f"Started at: {datetime.datetime.now()}")
    print(f"Arguments: {vars(args)}")
    
    try:
        results = evaluate_model(args)
        print(f"\nEvaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())