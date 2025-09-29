"""
Increment functions for explicit ODE schemes.

This module contains increment functions that compute dy for advancing
the solution from y(t) to y(t+dt) in explicit ODE schemes.
"""

from typing import Callable, Dict, Type
import torch
import torch.nn as nn

class Euler(nn.Module):
    """
    Euler increment function: dy = f(t, y)
    
    The simplest explicit scheme where the increment is just the derivative
    at the current point.
    """
    order = 1
    name = 'euler'

    def forward(
        self, 
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        y: torch.Tensor, 
        t: torch.Tensor, 
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Euler increment.
        
        Args:
            func: ODE function f(t, y)
            y: Current solution value
            t: Current time
            dt: Time step (unused in Euler, but kept for interface consistency)
            
        Returns:
            dy: Increment dy = f(t, y)
        """
        return func(t, y)


class RK4(nn.Module):
    """
    4th-order Runge-Kutta increment function.
    
    Computes the weighted average of four slope estimates to achieve
    4th-order accuracy.
    """
    order = 4
    name = 'rk4'

    def forward(
        self, 
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        y: torch.Tensor, 
        t: torch.Tensor, 
        dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RK4 increment.
        
        Args:
            func: ODE function f(t, y)
            y: Current solution value
            t: Current time
            dt: Time step
            
        Returns:
            dy: RK4 increment dy = (k1 + 2*k2 + 2*k3 + k4)/6
        """
        half_dt = dt * 0.5
        
        # Four slope estimates
        k1 = func(t, y)
        k2 = func(t + half_dt, y + k1 * half_dt)
        k3 = func(t + half_dt, y + k2 * half_dt)
        k4 = func(t + dt, y + k3 * dt)
        
        # Weighted average with precomputed constant
        return (k1 + 2 * (k2 + k3) + k4) * (1.0/6.0)


# Dictionary of available increment functions
INCREMENTS: Dict[str, Type[nn.Module]] = {
    'euler': Euler,
    'rk4': RK4,
}


def get_increment_function(method: str) -> Type[nn.Module]:
    """
    Get increment function by name.
    
    Args:
        method: String name of the method ('euler', 'rk4')
        
    Returns:
        Increment function class
        
    Raises:
        KeyError: If method is not recognized
    """
    if method not in INCREMENTS:
        available = ', '.join(INCREMENTS.keys())
        raise KeyError(f"Unknown increment method '{method}'. Available: {available}")
    
    return INCREMENTS[method]