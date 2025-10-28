"""
Test model definitions for performance regression tests.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random

# Add paths for imports
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(base_dir, "examples"))
sys.path.insert(0, os.path.join(base_dir, "experiments", "otflowlarge"))

from Phi import Phi


class SimpleODE(nn.Module):
    """Simple test ODE: y'(t) = A @ tanh(B @ y + b)"""

    def __init__(self, dim, seed=None):
        super().__init__()

        # Set deterministic seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        self.A = nn.Linear(dim, dim, bias=False)
        self.B = nn.Linear(dim, dim, bias=True)

        # Initialize parameters
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.zeros_(self.B.bias)
    
    def forward(self, t, y):
        return self.A(torch.tanh(self.B(y)))


class OTFlowODE(nn.Module):
    """ODE wrapper for Phi that handles (t, y) interface"""
    
    def __init__(self, phi):
        super().__init__()
        self.phi = phi
    
    def forward(self, t, y):
        # y is shape (batch, d)
        # need to concatenate time to make (batch, d+1)
        t_expanded = t.expand(y.shape[0], 1)
        yt = torch.cat([y, t_expanded], dim=1)
        
        # Call phi to get gradient
        result = self.phi.trHess(yt, justGrad=True)
        
        # When justGrad=True, it returns just the gradient
        if isinstance(result, tuple):
            grad = result[0]
        else:
            grad = result
        
        # Return gradient (shape should be (batch, d))
        return grad[:, :-1]  # Remove time component


def create_simple_ode_model(dim=32, seed=42):
    """Create a simple ODE model for testing"""
    # Set deterministic seeds for reproducible model creation
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return SimpleODE(dim, seed=seed)


def create_otflow_model(d=256, m=128, nt=8, seed=42):
    """Create OTFlow model for complex testing"""
    # Set deterministic seeds for reproducible model creation
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    phi = Phi(nTh=nt, d=d, m=m, alph=[1.0, 100.0, 15.0])
    return OTFlowODE(phi)


def create_test_data(model_type, device='cuda:0', seed=42):
    """Create test data for different model types"""
    # Set deterministic seeds for reproducible data creation
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if model_type == 'simple':
        # Simple ODE test data - match the model dimensions
        batch_size = 64
        dim = 32  # This should match the model dim
        nt = 14

        x = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
        t = torch.linspace(0, 1, nt, device=device)

        return x, t

    elif model_type == 'otflow':
        # OTFlow test data
        batch_size = 128
        dim = 256
        nt = 8

        x = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
        t = torch.linspace(0, 1, nt, device=device)

        return x, t

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model_type):
    """Get model information for reporting"""
    if model_type == 'simple':
        return {
            'name': 'Simple ODE',
            'description': 'y\'(t) = A @ tanh(B @ y + b)',
            'dim': 32,
            'complexity': 'Low'
        }
    elif model_type == 'otflow':
        return {
            'name': 'OTFlow',
            'description': 'Optimal Transport Flow with Phi neural network',
            'dim': 256,
            'complexity': 'High'
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")