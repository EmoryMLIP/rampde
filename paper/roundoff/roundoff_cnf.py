"""
Roundoff error analysis for Continuous Normalizing Flows (CNF).

Uses exact hyperparameters from experiments/cnf/cnf.py:
- hidden_dim = 32
- width = 128
- Dataset: 8gaussians (single batch)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cnf'))

from roundoff_analyzer import RoundoffAnalyzer
import toy_data


class HyperNetwork(nn.Module):
    """Hyper-network for producing time-dependent weights."""
    
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
        t = t.view(1, 1)
        params = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(t)))))
        
        # params is [1, 3*blocksize + width], need to squeeze batch dimension
        params = params.squeeze(0)
        
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)
        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        
        return W, B, U


def hyper_trace(W, B, U, x, target_dtype):
    """Compute the trace of the Jacobian using the Hutchinson estimator."""
    W = W.to(target_dtype)  # [w, d, 1]
    B = B.to(target_dtype)  # [w, 1, 1]
    U = U.to(target_dtype)  # [w, 1, d]
    x = x.to(target_dtype)  # [n, d]

    w, d, _ = W.shape
    n = x.shape[0]
    x_exp = x.unsqueeze(0).expand(w, -1, -1)  # [w, n, d]

    # s_j = x @ w_j + b_j
    s = torch.bmm(x_exp, W).squeeze(-1)       # [w, n]
    
    # For numerical stability, compute in float32 when target is lower precision
    if target_dtype in [torch.float16, torch.bfloat16]:
        s = s.to(target_dtype) + B.to(target_dtype).squeeze(-1)  # [w, n]
        deriv = 1 - torch.tanh(s)**2  # [w, n] - stays in float32
    else:
        s = s + B.squeeze(-1)  # [w, n]
        deriv = 1 - torch.tanh(s)**2  # [w, n]

    # u_j * w_j 
    uw_dot = torch.bmm(U, W).squeeze(-1).squeeze(-1)  # [w]
    uw_dot = uw_dot.view(w, 1)  # [w, 1]

    # For numerical stability
    if target_dtype in [torch.float16, torch.bfloat16]:
        trace_all = deriv * uw_dot.to(torch.float32)  # [w, n]
        trace_sum = trace_all.sum(dim=0) 
        trace_est = trace_sum / w  # [n]
        return trace_est.to(target_dtype)
    else:
        trace_all = deriv * uw_dot  # [w, n]
        trace_sum = trace_all.sum(dim=0) 
        trace_est = trace_sum / w  # [n]
        return trace_est


class CNF(nn.Module):
    """Continuous Normalizing Flow model."""
    
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)
        # Store init args for recreation
        self._init_args = (in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        
        Z = z.unsqueeze(0).repeat(self.width, 1, 1)
        W, B, U = self.hyper_net(t)
        
        # Compute drift
        h = torch.tanh(torch.bmm(Z, W) + B)  # [width, batch, 1]
        F = torch.bmm(h, U).mean(0)  # [batch, 2]
        
        # Compute trace for density update
        target_dtype = z.dtype
        Jac_trace = hyper_trace(W, B, U, z, target_dtype)
        
        return (F, -Jac_trace.view(-1, 1))


class CNFRoundoffAnalyzer(RoundoffAnalyzer):
    """Roundoff analyzer specific to CNF experiments."""
    
    def __init__(self, device='cuda'):
        super().__init__('cnf', device)
        
    def compute_loss(self, sol):
        """Compute negative log-likelihood loss for CNF."""
        if isinstance(sol, tuple):
            # sol is (z_all, logp_z_all) where each has shape [n_times, batch, ...]
            z_all, logp_z_all = sol
            z_t = z_all[-1]  # Final time: [batch, 2]
            logp_z_t = logp_z_all[-1]  # Final time: [batch, 1]
        else:
            # This shouldn't happen for CNF, but handle it gracefully
            return super().compute_loss(sol)
            
        # Standard Gaussian log probability
        logp_x = torch.sum(-0.5 * z_t**2 - 0.5 * np.log(2 * np.pi), dim=1)
        logp = logp_x + logp_z_t.view(-1)  # Use view instead of squeeze for safety
        return -logp.mean()  # Negative log-likelihood


def main():
    """Run CNF roundoff experiment."""
    import argparse

    parser = argparse.ArgumentParser(description='CNF Roundoff Error Analysis')
    parser.add_argument('--checkpoint', type=str,
                       default='/local/scratch/lruthot/code/rampde/paper/cnf/raw_data/cnf/2spirals_float32_torchdiffeq_rk4_lr_0.01_niters_2000_num_samples_1024_hidden_dim_32_width_128_num_timesteps_128_seed42_20251025_084625/ckpt.pth',
                       help='Path to trained model checkpoint (fp32)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for data generation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analyzer = CNFRoundoffAnalyzer(device)

    # Setup model with exact hyperparameters
    func = CNF(in_out_dim=2, hidden_dim=32, width=128).to(device)

    # Load trained weights
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    func.load_state_dict(checkpoint['func_state_dict'])
    print(f"Loaded trained weights from checkpoint at iteration {checkpoint['iteration']}")
    print(f"Checkpoint path: {checkpoint_path}")

    # Setup data - single batch of 2spirals (matching the trained model)
    data = toy_data.inf_train_gen('2spirals', batch_size=args.batch_size)
    x = torch.from_numpy(data).to(device).to(torch.float32)

    # Initial state: data points with zero log-density
    logp_z0 = torch.zeros(x.shape[0], 1).to(device).to(torch.float32)
    y0 = (x, logp_z0)
    
    # Experiment configuration
    # Base timesteps = 128 (default for CNF)
    # Test with 4x fewer (32) and 8x more (1024)
    timesteps_values = [64, 128, 256, 512, 1024, 2048, 4096]
    methods = ['euler', 'rk4']
    precisions = ['float16', 'bfloat16']
    
    # Scaler configurations: (odeint_type, scaler_type)
    scaler_configs = [
        # BF16 configurations (no scaling needed)
        ('torchdiffeq', None),        # torchdiffeq + bf16
        ('rampde', None),        # rampde + bf16
        
        # FP16 configurations (only when precision is float16)
        ('torchdiffeq', 'none'),      # torchdiffeq + fp16 + no_scaling
        ('torchdiffeq', 'grad'),      # torchdiffeq + fp16 + grad_scaling
        ('rampde', 'none'),      # rampde + fp16 + no_scaling
        ('rampde', 'grad'),      # rampde + fp16 + grad_scaling
        ('rampde', 'dynamic'),   # rampde + fp16 + dynamic_scaling
    ]
    
    print("Starting CNF roundoff experiment...")
    print(f"Model: hidden_dim=32, width=128 (loaded from trained checkpoint)")
    print(f"Data: 2spirals, batch_size=128")
    print(f"Timesteps: {timesteps_values}")
    
    # Run experiment
    analyzer.run_experiment(
        func=func,
        y0=y0,
        timesteps_values=timesteps_values,
        methods=methods,
        precisions=precisions,
        odeint_types=['torchdiffeq', 'rampde'],  # Will be filtered by scaler_configs
        scaler_configs=scaler_configs
    )
    
    # Save results to raw_data directory
    output_dir = os.path.join(os.path.dirname(__file__), 'raw_data')
    analyzer.save_results(output_dir)
    

if __name__ == '__main__':
    main()