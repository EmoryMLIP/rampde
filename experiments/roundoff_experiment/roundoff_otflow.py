"""
Roundoff error analysis for OTFlow (Optimal Transport Flow).

Single-configuration script for reproducible SLURM-based execution.
Uses exact hyperparameters from experiments/otflowlarge/otflowlarge.py:
- hidden_dim (m) = 1024
- alpha = [1.0, 2000.0, 800.0]
- Dataset: BSDS300 (deterministically sampled batch)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import pandas as pd
import fcntl
from torch.nn.functional import pad

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'otflowlarge'))

from roundoff_analyzer import RoundoffAnalyzer
import datasets
from Phi import Phi


class OTFlow(nn.Module):
    """OTFlow model matching otflowlarge.py implementation."""
    
    def __init__(self, in_out_dim, hidden_dim, alpha=[1.0]*2, Phi_class=None):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        if Phi_class is None:
            raise ValueError("Phi_class must be provided")
        self.Phi = Phi_class(2, hidden_dim, in_out_dim, alph=alpha)
        # Store init args for recreation
        self._init_args = (in_out_dim, hidden_dim, alpha, Phi_class)

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

        return (dz_dt, dlogp_dt, cost_L_dt, cost_HJB_dt)


def load_data(data_name, datasets_module):
    """Load data using the datasets module."""
    if data_name == "miniboone":
        data = datasets_module.MINIBOONE()
    elif data_name == "bsds300":
        data = datasets_module.BSDS300()
    elif data_name == "power":
        data = datasets_module.POWER()
    elif data_name == "gas":
        data = datasets_module.GAS()
    elif data_name == "hepmass":
        data = datasets_module.HEPMASS()
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return data


class OTFlowRoundoffAnalyzer(RoundoffAnalyzer):
    """Roundoff analyzer specific to OTFlow experiments."""
    
    def __init__(self, alpha, device='cuda'):
        super().__init__('otflow', device)
        self.alpha = alpha
        
    def compute_loss(self, sol):
        """Compute OTFlow loss matching otflowlarge.py."""
        # sol contains (z, logp, L, H) at final time
        # Handle both tuple and tensor outputs
        if isinstance(sol, tuple):
            # ODE returns tuple of tensors, each of shape [n_times, batch, ...]
            z_all, pl_all, L_all, H_all = sol
            z1 = z_all[-1]
            pl1 = pl_all[-1]
            L1 = L_all[-1]
            H1 = H_all[-1]
        else:
            # This shouldn't happen for OTFlow
            return super().compute_loss(sol)
        
        # Standard Gaussian for p_z0
        d = z1.shape[1]
        cov = torch.eye(d, device=z1.device) * 0.1
        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d, device=z1.device),
            covariance_matrix=cov
        )
        
        # Log probability
        logp_x = p_z0.log_prob(z1).view(-1, 1) + pl1
        
        # Total loss with alpha weighting
        loss = (-self.alpha[2] * logp_x.mean() + 
                self.alpha[0] * L1.mean() + 
                self.alpha[1] * H1.mean())
        
        return loss


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seeded all RNGs with seed={seed}")


def get_deterministic_data_batch(train_x, batch_size: int, seed: int):
    """Get a deterministic batch of data using fixed indexing."""
    # Use seed to create deterministic indices
    torch.manual_seed(seed)  # Ensure consistent indexing
    total_samples = train_x.size(0)
    
    # Create deterministic indices - use first batch_size samples with seed offset
    start_idx = seed % (total_samples - batch_size)
    indices = torch.arange(start_idx, start_idx + batch_size)
    
    return train_x[indices]


def check_existing_result(csv_file: str, config: dict) -> bool:
    """Check if this configuration already exists in the CSV file."""
    if not os.path.exists(csv_file):
        return False
    
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return False
        
        # Check for matching configuration
        mask = (
            (df['precision'] == config['precision']) &
            (df['odeint_type'] == config['odeint_type']) &
            (df['method'] == config['method']) &
            (df['n_timesteps'] == config['n_timesteps'])
        )
        
        # Handle scaler_type which can be None/NaN
        if config['scaler_type'] is None:
            mask = mask & (df['scaler_type'].isna())
        else:
            mask = mask & (df['scaler_type'] == config['scaler_type'])
        
        return mask.any()
    except Exception as e:
        print(f"Warning: Could not check existing results: {e}")
        return False


def append_result_to_csv(result: dict, csv_file: str):
    """Append a single result to CSV file with file locking."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Convert result to DataFrame
    df_new = pd.DataFrame([result])
    
    # Use file locking to prevent race conditions
    with open(csv_file, 'a') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            # Check if file is empty (need headers)
            file_exists_and_not_empty = os.path.getsize(csv_file) > 0
            
            # Write to CSV
            df_new.to_csv(f, header=not file_exists_and_not_empty, index=False)
            f.flush()
            
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    print(f"✅ Result appended to {csv_file}")


def parse_arguments():
    """Parse command line arguments for single configuration."""
    parser = argparse.ArgumentParser(
        description="Run OTFlow roundoff experiment for a single configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--precision', type=str, required=True,
                       choices=['float16', 'bfloat16'],
                       help='Precision to use')
    
    parser.add_argument('--odeint_type', type=str, required=True,
                       choices=['torchmpnode', 'torchdiffeq'],
                       help='ODE solver type')
    
    parser.add_argument('--scaler_type', type=str, default=None,
                       choices=['none', 'grad', 'dynamic'],
                       help='Scaler type (auto-determined if None)')
    
    parser.add_argument('--method', type=str, required=True,
                       choices=['euler', 'rk4'],
                       help='Integration method')
    
    parser.add_argument('--n_timesteps', type=int, required=True,
                       choices=[8, 16, 32, 64, 128, 256],
                       help='Number of timesteps')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for analysis')
    
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip if configuration already exists in CSV')
    
    return parser.parse_args()


def validate_configuration(args):
    """Validate argument combinations and auto-determine scaler_type if needed."""
    # Auto-determine scaler_type based on precision and odeint_type
    if args.scaler_type is None:
        if args.precision == 'bfloat16':
            args.scaler_type = None  # bfloat16 doesn't need scaling
        elif args.precision == 'float16':
            if args.odeint_type == 'torchmpnode':
                args.scaler_type = 'dynamic'  # default for torchmpnode
            else:
                args.scaler_type = 'grad'  # default for torchdiffeq
    
    # Validate scaler_type combinations
    if args.precision == 'bfloat16' and args.scaler_type is not None:
        raise ValueError(f"bfloat16 should not use scaling, but got scaler_type={args.scaler_type}")
    
    if args.precision == 'float16' and args.scaler_type is None:
        raise ValueError(f"float16 requires explicit scaler_type, but got None")
    
    if args.odeint_type == 'torchdiffeq' and args.scaler_type == 'dynamic':
        raise ValueError(f"torchdiffeq doesn't support dynamic scaling")
    
    print(f"✅ Configuration validated: {args.precision} + {args.odeint_type} + {args.scaler_type} + {args.method} + {args.n_timesteps} timesteps")
    return args


def main():
    """Run OTFlow roundoff experiment for a single configuration."""
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_configuration(args)
    
    # Seed everything for reproducibility
    seed_everything(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create configuration dict for tracking
    config = {
        'precision': args.precision,
        'odeint_type': args.odeint_type,
        'scaler_type': args.scaler_type,
        'method': args.method,
        'n_timesteps': args.n_timesteps
    }
    
    # Check if this configuration already exists
    csv_file = os.path.join(args.output_dir, 'otflow_roundoff_results.csv')
    if args.skip_existing and check_existing_result(csv_file, config):
        print(f"⚠️  Configuration already exists in {csv_file}, skipping...")
        return
    
    print(f"🎯 Running single configuration:")
    print(f"   precision={args.precision}, odeint_type={args.odeint_type}")
    print(f"   scaler_type={args.scaler_type}, method={args.method}")
    print(f"   n_timesteps={args.n_timesteps}, seed={args.seed}")
    
    # Load BSDS300 data
    data = load_data('bsds300', datasets)
    train_x = torch.from_numpy(data.trn.x).float().to(device)
    d = train_x.size(1)
    
    print(f"📊 Loaded BSDS300: dimension={d}, train_size={train_x.shape[0]}")
    
    # Setup model with exact hyperparameters
    alpha = [1.0, 2000.0, 800.0]
    analyzer = OTFlowRoundoffAnalyzer(alpha, device)
    func = OTFlow(in_out_dim=d, hidden_dim=1024, alpha=alpha, Phi_class=Phi).to(device)
    
    # Load trained weights
    checkpoint_path = '/local/scratch/lruthot/code/torchmpnode/experiments/results_paper/otflowlarge/bsds300_float32_torchmpnode_rk4_alpha_1.0,2000.0,800.0_lr_0.001_niters_10000_batch_size_512_hidden_dim_1024_nt_16_seed42_20250725_165512/ckpt_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    func.load_state_dict(checkpoint['model_state_dict'])
    print(f"🏋️  Loaded trained weights from checkpoint at iteration {checkpoint['iteration']}")
    
    # Setup deterministic data batch
    x = get_deterministic_data_batch(train_x, args.batch_size, args.seed)
    print(f"🎲 Using deterministic batch: batch_size={args.batch_size}, seed={args.seed}")
    
    # Initial state: data points with zero log-density and costs
    logp0 = torch.zeros(args.batch_size, 1).to(device)
    L0 = torch.zeros(args.batch_size, 1).to(device)
    H0 = torch.zeros(args.batch_size, 1).to(device)
    y0 = (x, logp0, L0, H0)
    
    print(f"🚀 Starting single configuration analysis...")
    
    try:
        # Run single configuration
        result = analyzer.run_single_configuration(
            func=func,
            y0=y0,
            n_timesteps=args.n_timesteps,
            method=args.method,
            precision=args.precision,
            odeint_type=args.odeint_type,
            scaler_type=args.scaler_type
        )
        
        # Print summary
        print(f"✅ SUCCESS!")
        print(f"   Deterministic: {result['is_deterministic']}")
        print(f"   Solution error: {result['sol_error_mean']:.2e}")
        if result['grad_y0_error_mean'] != float('nan'):
            print(f"   Gradient error: {result['grad_y0_error_mean']:.2e}")
        
        # Append result to CSV
        append_result_to_csv(result, csv_file)
        print(f"📝 Result logged to CSV")
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        
        # Store failed result
        failed_result = {
            'experiment': 'otflow',
            'n_timesteps': args.n_timesteps,
            'h': 1.0 / args.n_timesteps,
            'method': args.method,
            'precision': args.precision,
            'odeint_type': args.odeint_type,
            'scaler_type': args.scaler_type,
            'error': str(e)
        }
        
        # Append failed result to CSV
        append_result_to_csv(failed_result, csv_file)
        print(f"📝 Error logged to CSV")
        
        # Re-raise for SLURM job failure detection
        raise
    

if __name__ == '__main__':
    main()