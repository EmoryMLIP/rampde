"""
Core roundoff analysis module for measuring precision-related errors in ODE integration.

This module provides a base class for running roundoff error experiments that:
1. Compare low-precision (fp16/bf16) results against fp64 reference
2. Check for determinism and run multiple iterations if needed
3. Measure errors as step size h approaches zero
"""

import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
import os
import sys
import copy

# Type alias for tensor or tuple of tensors
TupleOrTensor = Union[torch.Tensor, Tuple[torch.Tensor, ...]]

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add rampde root directory for rampde imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import setup_environment, get_precision_dtype


class RoundoffAnalyzer:
    """Base class for roundoff error analysis experiments."""
    
    def __init__(self, experiment_name: str, device: str = 'cuda'):
        self.experiment_name = experiment_name
        self.device = torch.device(device)
        self.results = []
        
    def check_determinism(
        self, 
        func: nn.Module,
        y0: TupleOrTensor,
        t: torch.Tensor,
        method: str,
        dtype: torch.dtype,
        odeint_fn: Callable,
        scaler: Optional[object] = None,
        odeint_type: str = 'torchdiffeq',
        scaler_type: Optional[str] = None
    ) -> Tuple[bool, TupleOrTensor, Dict[str, torch.Tensor]]:
        """
        Check if ODE integration is deterministic by running twice.
        
        Returns:
            (is_deterministic, solution, gradients_dict)
        """
        def run_once():
            # Clone inputs and reset gradients
            # Try deepcopy, but if it fails (e.g., JIT functions), recreate manually
            try:
                func_copy = copy.deepcopy(func)
            except:
                # For modules that can't be pickled, create new instance
                func_copy = type(func)(*getattr(func, '_init_args', ()))
                func_copy.load_state_dict(func.state_dict())
                func_copy = func_copy.to(self.device)
            
            # Handle tuple inputs - keep in fp32 for autocast to work properly
            if isinstance(y0, tuple):
                y0_copy = tuple(y.clone().detach().requires_grad_(True) for y in y0)
            else:
                y0_copy = y0.clone().detach().requires_grad_(True)
            
            # Keep time tensor in fp32 for autocast
            t_copy = t.to(torch.float32)
            
            # Clear any existing gradients - keep parameters in float32, autocast will handle conversion
            for p in func_copy.parameters():
                p.grad = None
            
            # Run forward pass
            if odeint_type == 'torchdiffeq' and scaler_type == 'grad' and dtype == torch.float16:
                # Special handling for torchdiffeq with GradScaler
                from torch.cuda.amp import GradScaler
                
                # Adaptive scaling: try reducing scale until gradients are finite
                max_attempts = 20  # More attempts to reach very small scales
                min_scale = 0.001  # Much smaller minimum scale
                initial_scale = 65536.0
                
                for attempt in range(max_attempts):
                    try:
                        # Create fresh GradScaler for each attempt (PyTorch requirement)
                        current_scale = initial_scale / (2 ** attempt)
                        if current_scale < min_scale:
                            current_scale = min_scale
                        grad_scaler = GradScaler(init_scale=current_scale)
                        
                        # Clear any existing gradients
                        for p in func_copy.parameters():
                            p.grad = None
                        if isinstance(y0_copy, tuple):
                            for y in y0_copy:
                                y.grad = None
                        else:
                            y0_copy.grad = None
                        
                        with autocast(device_type='cuda', dtype=dtype):
                            sol = odeint_fn(func_copy, y0_copy, t_copy, method=method)
                            loss = self.compute_loss(sol)
                        
                        # Scaled backward pass
                        grad_scaler.scale(loss).backward()
                        
                        # Create dummy optimizer to unscale gradients
                        params = list(func_copy.parameters())
                        if isinstance(y0_copy, tuple):
                            params.extend(list(y0_copy))
                        else:
                            params.append(y0_copy)
                        dummy_optimizer = torch.optim.SGD(params, lr=0.1)
                        grad_scaler.unscale_(dummy_optimizer)
                        
                        # Check if gradients are finite and verify dtypes
                        gradients_finite = True
                        max_grad_norm = 0.0
                        for i, p in enumerate(func_copy.parameters()):
                            if p.grad is not None:
                                grad_norm = torch.norm(p.grad).item()
                                max_grad_norm = max(max_grad_norm, grad_norm)
                                if attempt == 0:  # Only print on first attempt to avoid spam
                                    print(f"    Parameter {i}: weight dtype={p.dtype}, grad dtype={p.grad.dtype}, grad_norm={grad_norm:.2e}")
                                if not torch.isfinite(p.grad).all():
                                    gradients_finite = False
                                    break
                        
                        # Show gradient magnitude progress
                        if attempt > 0 or not gradients_finite:
                            print(f"    Attempt {attempt + 1} (scale={current_scale}): max_grad_norm={max_grad_norm:.2e}, finite={gradients_finite}")
                        
                        if gradients_finite:
                            # Success! Store final scale for reporting
                            final_scale = grad_scaler.get_scale()
                            if attempt > 0:
                                print(f"  Gradient scaling succeeded after {attempt + 1} attempts (final_scale={final_scale})")
                            break
                        else:
                            raise RuntimeError("Gradients not finite after unscaling")
                            
                    except Exception as e:
                        if current_scale <= min_scale or attempt == max_attempts - 1:
                            final_scale = current_scale
                            raise RuntimeError(f"torchdiffeq gradient scaling failed after {max_attempts} attempts (final_scale={final_scale}): {str(e)}")
                        
                        if attempt < max_attempts - 1:
                            next_scale = initial_scale / (2 ** (attempt + 1))
                            if next_scale < min_scale:
                                next_scale = min_scale
                            print(f"  Attempt {attempt + 1} failed (scale={current_scale}), trying scale={next_scale}: {str(e)}")
                        else:
                            print(f"  All attempts failed. Reached minimum scale {min_scale}")
            else:
                # Standard forward pass
                with autocast(device_type='cuda', dtype=dtype):
                    if scaler is not None and scaler is not False:
                        if odeint_type == 'rampde':
                            sol = odeint_fn(func_copy, y0_copy, t_copy, method=method, loss_scaler=scaler)
                        else:
                            sol = odeint_fn(func_copy, y0_copy, t_copy, method=method)
                    elif scaler is False and odeint_type == 'rampde':
                        # Explicitly disable scaling for rampde
                        sol = odeint_fn(func_copy, y0_copy, t_copy, method=method, loss_scaler=False)
                    else:
                        sol = odeint_fn(func_copy, y0_copy, t_copy, method=method)
                    loss = self.compute_loss(sol)
                
                # Backward pass
                loss.backward()
            
            # Extract gradients
            if isinstance(y0_copy, tuple):
                y0_grads = tuple(y.grad.detach().clone() if y.grad is not None else None for y in y0_copy)
            else:
                y0_grads = y0_copy.grad.detach().clone() if y0_copy.grad is not None else None
                
            grads = {
                'y0': y0_grads,
                'params': {name: p.grad.detach().clone() if p.grad is not None else None 
                          for name, p in func_copy.named_parameters()}
            }
            
            # Handle tuple solutions
            if isinstance(sol, tuple):
                sol_detached = tuple(s.detach().clone() for s in sol)
            else:
                sol_detached = sol.detach().clone()
            
            return sol_detached, loss.detach().clone(), grads
        
        # Run twice
        sol1, loss1, grads1 = run_once()
        sol2, loss2, grads2 = run_once()
        
        # Check determinism
        is_deterministic = True
        
        # Check solution
        if isinstance(sol1, tuple):
            for s1, s2 in zip(sol1, sol2):
                if not torch.allclose(s1, s2, rtol=1e-7, atol=1e-8):
                    is_deterministic = False
                    break
        else:
            if not torch.allclose(sol1, sol2, rtol=1e-7, atol=1e-8):
                is_deterministic = False
            
        # Check loss
        if not torch.allclose(loss1, loss2, rtol=1e-7, atol=1e-8):
            is_deterministic = False
            
        # Check gradients
        if isinstance(grads1['y0'], tuple) and isinstance(grads2['y0'], tuple):
            for g1, g2 in zip(grads1['y0'], grads2['y0']):
                if g1 is not None and g2 is not None:
                    if not torch.allclose(g1, g2, rtol=1e-7, atol=1e-8):
                        is_deterministic = False
                        break
        elif grads1['y0'] is not None and grads2['y0'] is not None:
            if not torch.allclose(grads1['y0'], grads2['y0'], rtol=1e-7, atol=1e-8):
                is_deterministic = False
                
        for name in grads1['params']:
            if grads1['params'][name] is not None and grads2['params'][name] is not None:
                if not torch.allclose(grads1['params'][name], grads2['params'][name], rtol=1e-7, atol=1e-8):
                    is_deterministic = False
                    break
        
        return is_deterministic, sol1, grads1
    
    def compute_loss(self, sol: TupleOrTensor) -> torch.Tensor:
        """Compute loss from ODE solution. Override in subclasses."""
        # Default: sum of final state
        if isinstance(sol, tuple):
            return sol[0][-1].sum()  # Use first element of tuple
        else:
            return sol[-1].sum()
    
    def run_single_configuration(
        self,
        func: nn.Module,
        y0: TupleOrTensor,
        n_timesteps: int,
        method: str,
        precision: str,
        odeint_type: str,
        scaler_type: Optional[str] = None,
        n_runs: int = 10
    ) -> Dict:
        """Run a single configuration and measure errors against fp64 reference."""
        
        # Setup time grid based on number of timesteps
        t_final = 1.0
        t = torch.linspace(0.0, t_final, n_timesteps + 1, device=self.device, dtype=torch.float64)
        
        # Get ODE integrator
        if odeint_type == 'rampde':
            from rampde import odeint
            from rampde.loss_scalers import DynamicScaler
        else:
            from torchdiffeq import odeint
            DynamicScaler = None
        
        # First, compute fp64 reference
        try:
            func_ref = copy.deepcopy(func)
        except:
            # For modules that can't be pickled, create new instance
            func_ref = type(func)(*getattr(func, '_init_args', ()))
            func_ref.load_state_dict(func.state_dict())
            func_ref = func_ref.to(self.device)
        
        # Handle tuple inputs for fp64 reference
        if isinstance(y0, tuple):
            y0_ref = tuple(y.clone().detach().to(torch.float64).requires_grad_(True) for y in y0)
        else:
            y0_ref = y0.clone().detach().to(torch.float64).requires_grad_(True)
        
        # Clear gradients and convert parameters to float64 for reference
        for p in func_ref.parameters():
            p.data = p.data.to(torch.float64)
            p.grad = None
        
        # Run reference in fp64
        # For torchdiffeq, it preserves float64. For rampde, we need to handle the conversion
        sol_ref = odeint(func_ref, y0_ref, t, method=method)
        
        # Handle tuple outputs (e.g., CNF returns (z, logp))
        if isinstance(sol_ref, tuple):
            # Convert tuple elements to float64 if needed
            sol_ref_64 = tuple(s.to(torch.float64) if s.dtype != torch.float64 else s for s in sol_ref)
            loss_ref = self.compute_loss(sol_ref_64)
        else:
            # Ensure solution is in float64 for loss computation
            if sol_ref.dtype != torch.float64:
                sol_ref_64 = sol_ref.to(torch.float64)
                sol_ref_64.requires_grad_(True)
                loss_ref = self.compute_loss(sol_ref_64)
            else:
                loss_ref = self.compute_loss(sol_ref)
        
        loss_ref.backward()
        
        # Store reference values
        if isinstance(sol_ref, tuple):
            sol_ref_val = tuple(s.detach().clone() for s in sol_ref)
        else:
            sol_ref_val = sol_ref.detach().clone()
        
        # Handle tuple inputs for gradients
        if isinstance(y0, tuple):
            grad_y0_ref = tuple(y.grad.detach().clone() if y.grad is not None else None for y in y0_ref) if isinstance(y0_ref, tuple) else None
        else:
            grad_y0_ref = y0_ref.grad.detach().clone() if y0_ref.grad is not None else None
        grad_params_ref = {name: p.grad.detach().clone() if p.grad is not None else None 
                          for name, p in func_ref.named_parameters()}
        
        # Now run in lower precision
        dtype = get_precision_dtype(precision)
        
        # Setup scaler if needed
        scaler = None
        if precision == 'float16':
            if odeint_type == 'rampde':
                if scaler_type == 'dynamic' and DynamicScaler is not None:
                    scaler = DynamicScaler(dtype)
                elif scaler_type == 'grad':
                    # For rampde, grad scaling uses default (DynamicScaler)
                    scaler = None  # This triggers default DynamicScaler
                elif scaler_type == 'none':
                    scaler = False  # Explicitly disable scaling
            elif odeint_type == 'torchdiffeq' and scaler_type == 'grad':
                # For torchdiffeq, we'll use PyTorch's GradScaler
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
        
        # Check determinism
        try:
            func_test = copy.deepcopy(func)
        except:
            func_test = type(func)(*getattr(func, '_init_args', ()))
            func_test.load_state_dict(func.state_dict())
            func_test = func_test.to(self.device)
        
        # Handle tuple inputs for test
        if isinstance(y0, tuple):
            y0_test = tuple(y.clone().detach().requires_grad_(True) for y in y0)
        else:
            y0_test = y0.clone().detach().requires_grad_(True)
        
        # Keep model parameters in float32, autocast will handle conversion
        
        is_deterministic, sol_test, grads_test = self.check_determinism(
            func_test, y0_test, t.to(torch.float32), method, dtype, odeint, scaler, odeint_type, scaler_type
        )
        
        # If deterministic, we only need one run. Otherwise, run multiple times
        if is_deterministic:
            n_actual_runs = 1
            sols = [sol_test]
            grad_y0s = [grads_test['y0']]
            grad_params = [{name: grads_test['params'][name] for name in grads_test['params']}]
        else:
            n_actual_runs = n_runs
            sols = []
            grad_y0s = []
            grad_params = []
            
            for _ in range(n_runs):
                try:
                    func_run = copy.deepcopy(func)
                except:
                    func_run = type(func)(*getattr(func, '_init_args', ()))
                    func_run.load_state_dict(func.state_dict())
                    func_run = func_run.to(self.device)
                
                # Handle tuple inputs for each run - keep in fp32 for autocast
                if isinstance(y0, tuple):
                    y0_run = tuple(y.clone().detach().requires_grad_(True) for y in y0)
                else:
                    y0_run = y0.clone().detach().requires_grad_(True)
                
                # Keep time tensor in fp32 for autocast
                t_run = t.to(torch.float32)
                
                # Clear gradients
                for p in func_run.parameters():
                    p.grad = None
                
                if odeint_type == 'torchdiffeq' and scaler_type == 'grad' and dtype == torch.float16:
                    # Special handling for torchdiffeq with GradScaler (with adaptive scaling)
                    from torch.cuda.amp import GradScaler
                    
                    # Adaptive scaling: try reducing scale until gradients are finite
                    max_attempts = 20  # More attempts to reach very small scales
                    min_scale = 0.001  # Much smaller minimum scale
                    initial_scale = 65536.0
                    
                    for attempt in range(max_attempts):
                        try:
                            # Create fresh GradScaler for each attempt (PyTorch requirement)
                            current_scale = initial_scale / (2 ** attempt)
                            if current_scale < min_scale:
                                current_scale = min_scale
                            grad_scaler = GradScaler(init_scale=current_scale)
                            
                            # Clear any existing gradients
                            for p in func_run.parameters():
                                p.grad = None
                            if isinstance(y0_run, tuple):
                                for y in y0_run:
                                    y.grad = None
                            else:
                                y0_run.grad = None
                            
                            with autocast(device_type='cuda', dtype=dtype):
                                sol_run = odeint(func_run, y0_run, t_run, method=method)
                                loss_run = self.compute_loss(sol_run)
                            
                            # Scaled backward pass
                            grad_scaler.scale(loss_run).backward()
                            
                            # Create dummy optimizer to unscale gradients
                            params = list(func_run.parameters())
                            if isinstance(y0_run, tuple):
                                params.extend(list(y0_run))
                            else:
                                params.append(y0_run)
                            dummy_optimizer = torch.optim.SGD(params, lr=0.1)
                            grad_scaler.unscale_(dummy_optimizer)
                            
                            # Check if gradients are finite
                            gradients_finite = True
                            for p in func_run.parameters():
                                if p.grad is not None and not torch.isfinite(p.grad).all():
                                    gradients_finite = False
                                    break
                            
                            if gradients_finite:
                                # Success!
                                break
                            else:
                                raise RuntimeError("Gradients not finite after unscaling")
                                
                        except Exception as e:
                            if current_scale <= min_scale or attempt == max_attempts - 1:
                                # Final attempt failed, propagate error
                                raise e
                else:
                    with autocast(device_type='cuda', dtype=dtype):
                        if scaler is not None and scaler is not False:
                            if odeint_type == 'rampde':
                                sol_run = odeint(func_run, y0_run, t_run, method=method, 
                                               loss_scaler=scaler)
                            else:
                                sol_run = odeint(func_run, y0_run, t_run, method=method)
                        elif scaler is False and odeint_type == 'rampde':
                            # Explicitly disable scaling for rampde
                            sol_run = odeint(func_run, y0_run, t_run, method=method, 
                                           loss_scaler=False)
                        else:
                            sol_run = odeint(func_run, y0_run, t_run, method=method)
                        loss_run = self.compute_loss(sol_run)
                    
                    loss_run.backward()
                
                sols.append(sol_run.detach() if not isinstance(sol_run, tuple) else sol_run)
                
                # Handle tuple gradients
                if isinstance(y0_run, tuple):
                    grad_y0s.append(tuple(y.grad.detach() if y.grad is not None else None for y in y0_run))
                else:
                    grad_y0s.append(y0_run.grad.detach() if y0_run.grad is not None else None)
                    
                grad_params.append({name: p.grad.detach() if p.grad is not None else None 
                                  for name, p in func_run.named_parameters()})
        
        # Compute errors
        def relative_error(x, x_ref):
            if x is None or x_ref is None:
                return float('nan')
            
            # Handle tuples
            if isinstance(x, tuple) and isinstance(x_ref, tuple):
                errors = []
                for xi, xi_ref in zip(x, x_ref):
                    if xi is not None and xi_ref is not None:
                        errors.append((torch.norm(xi.to(torch.float64) - xi_ref) / torch.norm(xi_ref)).item())
                return np.mean(errors) if errors else float('nan')
            else:
                return (torch.norm(x.to(torch.float64) - x_ref) / torch.norm(x_ref)).item()
        
        # Compute errors for each run
        sol_errors = []
        grad_y0_errors = []
        grad_param_errors = {name: [] for name in grad_params_ref}
        
        for i in range(n_actual_runs):
            # Solution error
            sol_errors.append(relative_error(sols[i], sol_ref_val))
            
            # Gradient errors
            if grad_y0s[i] is not None:
                grad_y0_errors.append(relative_error(grad_y0s[i], grad_y0_ref))
            
            for name in grad_params_ref:
                if name in grad_params[i] and grad_params[i][name] is not None:
                    grad_param_errors[name].append(
                        relative_error(grad_params[i][name], grad_params_ref[name])
                    )
        
        # Aggregate results
        result = {
            'experiment': self.experiment_name,
            'n_timesteps': n_timesteps,
            'h': 1.0 / n_timesteps,  # Keep h for backwards compatibility
            'method': method,
            'precision': precision,
            'odeint_type': odeint_type,
            'scaler_type': scaler_type,
            'is_deterministic': is_deterministic,
            'n_runs': n_actual_runs,
            'sol_error_mean': np.mean(sol_errors),
            'sol_error_std': np.std(sol_errors) if n_actual_runs > 1 else 0.0,
            'grad_y0_error_mean': np.mean(grad_y0_errors) if grad_y0_errors else float('nan'),
            'grad_y0_error_std': np.std(grad_y0_errors) if n_actual_runs > 1 and grad_y0_errors else 0.0,
        }
        
        # Add parameter gradient errors
        for name in grad_param_errors:
            if grad_param_errors[name]:
                result[f'grad_{name}_error_mean'] = np.mean(grad_param_errors[name])
                result[f'grad_{name}_error_std'] = np.std(grad_param_errors[name]) if n_actual_runs > 1 else 0.0
        
        return result
    
    def run_experiment(
        self,
        func: nn.Module,
        y0: TupleOrTensor,
        timesteps_values: List[int],
        methods: List[str],
        precisions: List[str],
        odeint_types: List[str],
        scaler_configs: List[Tuple[str, Optional[str]]]
    ):
        """Run full experiment across all configurations."""
        
        # Calculate actual number of configs after filtering
        total_configs = 0
        for precision in precisions:
            for odeint_type, scaler_type in scaler_configs:
                if precision == 'bfloat16' and scaler_type is not None:
                    continue
                if precision == 'float16' and scaler_type is None:
                    continue
                total_configs += 1
        total_configs *= len(timesteps_values) * len(methods)
        config_idx = 0
        
        for n_timesteps in timesteps_values:
            for method in methods:
                for precision in precisions:
                    for odeint_type, scaler_type in scaler_configs:
                        # Skip invalid configurations
                        if precision == 'bfloat16' and scaler_type is not None:
                            # BF16 doesn't need scaling, only run with None
                            continue
                        if precision == 'float16' and scaler_type is None:
                            # FP16 needs explicit scaling configuration
                            continue
                            
                        config_idx += 1
                        print(f"\n[{config_idx}/{total_configs}] Running n_timesteps={n_timesteps}, method={method}, "
                              f"precision={precision}, odeint={odeint_type}, scaler={scaler_type}")
                        
                        try:
                            result = self.run_single_configuration(
                                func, y0, n_timesteps, method, precision, odeint_type, scaler_type
                            )
                            self.results.append(result)
                            
                            # Print summary
                            print(f"  Deterministic: {result['is_deterministic']}")
                            print(f"  Solution error: {result['sol_error_mean']:.2e}")
                            if result['grad_y0_error_mean'] != float('nan'):
                                print(f"  Gradient error: {result['grad_y0_error_mean']:.2e}")
                            
                        except Exception as e:
                            print(f"  ERROR: {str(e)}")
                            # Store failed result
                            self.results.append({
                                'experiment': self.experiment_name,
                                'n_timesteps': n_timesteps,
                                'h': 1.0 / n_timesteps,
                                'method': method,
                                'precision': precision,
                                'odeint_type': odeint_type,
                                'scaler_type': scaler_type,
                                'error': str(e)
                            })
    
    def save_results(self, output_dir: str):
        """Save results to CSV file."""
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{self.experiment_name}_roundoff_results.csv')
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df