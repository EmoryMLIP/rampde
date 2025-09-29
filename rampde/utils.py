"""
Utility functions for rampde.

This module contains general utility functions used across the rampde package.
"""

from typing import Union
import torch


def _is_any_infinite(x: Union[torch.Tensor, tuple, list, None]) -> bool:
    """
    Recursively check if x (a tensor, list, or tuple of tensors) contains any non-finite values.
    
    This function handles nested structures of tensors and checks each tensor
    for the presence of infinite or NaN values. It's used throughout rampde
    for overflow detection in mixed precision computations.
    
    Args:
        x: Input to check - can be a tensor, list/tuple of tensors, or None
        
    Returns:
        True if any tensor element is inf or NaN; otherwise False
    """
    if x is None:
        return False
    if isinstance(x, torch.Tensor):
        return not x.isfinite().all().item()
    if isinstance(x, (list, tuple)):
        return any(_is_any_infinite(elem) for elem in x)
    # If x is neither, try to convert to tensor.
    try:
        return not torch.tensor(x).isfinite().all().item()
    except Exception:
        return False