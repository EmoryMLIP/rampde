"""Dynamic loss scaling for mixed precision training.

This module provides the DynamicScaler class for automatic loss scaling
in mixed precision training to prevent gradient underflow in float16 precision.
The scaler automatically adjusts the loss scale based on gradient magnitudes
to maintain numerical stability while maximizing precision utilization.
"""

from typing import Union, Optional, Any
import torch
import math
from .utils import _is_any_infinite



class DynamicScaler:
    """
    Dynamic loss scaler for mixed precision ODE training in float16.
    
    This class implements automatic loss scaling to prevent gradient underflow
    in float16 precision. It dynamically adjusts the scaling factor based on
    gradient magnitudes, increasing scale when gradients are small and 
    decreasing scale when overflow is detected.
    
    The scaling algorithm:
    1. Initialize scaling factor based on input tensor magnitude
    2. Scale gradients during backward pass to prevent underflow
    3. Check for overflow/underflow and adjust scaling factor
    4. Increase scale when gradients are small (< 0.5 * target)
    5. Decrease scale when overflow is detected
    
    Args:
        dtype_low: Low precision dtype (typically torch.float16)
        target_factor: Target gradient magnitude (default: 1.0 / eps)
        increase_factor: Factor to increase scale (default: 2.0)
        decrease_factor: Factor to decrease scale on overflow (default: 0.5)
        max_attempts: Maximum scaling attempts before giving up (default: 50)
        delta: Small value added for numerical stability (default: 0)
        
    Attributes:
        S: Current scaling factor (initialized on first use)
        is_initialized: Whether the scaler has been initialized
        eps: Machine epsilon for the low precision dtype
        target: Target gradient magnitude
        
    Example:
        >>> scaler = DynamicScaler(torch.float16)
        >>> # During forward pass, scaler.S is None until first backward
        >>> # During backward pass, scaler automatically initializes and adjusts
    """
    def __init__(
        self, 
        dtype_low: torch.dtype, 
        target_factor: Optional[float] = None, 
        increase_factor: float = 2.0, 
        decrease_factor: float = 0.5,
        max_attempts: int = 50, 
        delta: float = 0,
        verbose: bool = False
    ):
        self.dtype_low = dtype_low
        # Set a target norm if not provided: 1/epsilon for low precision.
        self.eps = torch.finfo(dtype_low).eps
        self.target = target_factor if target_factor is not None else 1.0 / self.eps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_attempts = max_attempts
        self.delta = delta
        self.is_initialized = False
        self.S: Optional[float] = None  # This will be initialized later
        self.__name__ = "DynamicScaler"
        self.verbose = verbose
        self.scale_history = []  # Track scale changes
    

    def init_scaling(self, a: torch.Tensor) -> None:
        """
        Initialize the scaling factor based on input tensor magnitude.
        
        This method analyzes the input tensor to determine an appropriate
        initial scaling factor. The scale is chosen to bring the tensor
        magnitude close to the target value while ensuring finite results.
        
        Args:
            a: Input tensor to analyze for scaling initialization
            
        Raises:
            ValueError: If input tensor contains non-finite or NaN values
            RuntimeError: If unable to find a finite scale after 20 attempts
        """
        if not(a.isfinite().all()) or a.isnan().any():
            raise ValueError("Input tensor contains non-finite or nan values.")
        
        # get the number of elements in a except for the 0th dimension
        target = self.target / math.sqrt(a.numel() / a.shape[0])
        a_max = a.abs().max()
        self.S = target / (a_max + self.delta).to(torch.float32)
        self.S = 2**(torch.round(torch.log2(self.S))).item()
        
        if self.verbose:
            print(f"\n[DynamicScaler] Initializing scale:")
            print(f"  - Input tensor shape: {a.shape}")
            print(f"  - Input max magnitude: {a_max.item():.6e}")
            print(f"  - Target magnitude: {target:.6e}")
            print(f"  - Initial scale S: {self.S:.6e}")
        
        # make sure S is a power of 2
        initial_S = self.S
        for i in range(20):         # 20 halvings = divide by 1 048 576
            anew = self.S * a
            if anew.isfinite().all():
                break
            self.S *= 0.5
            if self.verbose:
                print(f"  - Scale adjustment {i+1}: S reduced to {self.S:.6e}")
        else:
            raise RuntimeError(f"Scaler failed to find finite scale after 20 steps for {a.shape} with ||a||_inf = {a.abs().max()}.")
        
        if self.verbose and self.S != initial_S:
            print(f"  - Final scale S: {self.S:.6e}")
        
        self.is_initialized = True
        self.scale_history.append(('init', self.S))

    def update_on_overflow(self) -> None:
        """
        Update the scaling factor on overflow.
        
        Multiplies the current scaling factor by decrease_factor to reduce
        the scale and prevent further overflow in subsequent iterations.
        """
        old_S = self.S
        self.S *= self.decrease_factor
        if self.verbose:
            print(f"[DynamicScaler] Overflow detected: scale reduced from {old_S:.6e} to {self.S:.6e}")
        self.scale_history.append(('overflow', self.S))

    def check_for_increase(self, a: torch.Tensor) -> bool:
        """
        Check if scaling factor should be increased based on gradient magnitude.
        
        Compares the maximum absolute value of the tensor against the target
        magnitude to determine if the scale should be increased.
        
        Args:
            a: Tensor to check for magnitude
            
        Returns:
            True if the tensor magnitude is less than 50% of target, False otherwise
        """
        # Use .item() to return a Python bool, not a tensor
        a_max = a.abs().max()
        ratio = (a_max / self.target).item()
        should_increase = ratio < 0.5
        
        if self.verbose and should_increase:
            print(f"[DynamicScaler] Gradient magnitude check: max={a_max.item():.6e}, ratio={ratio:.6f} < 0.5, will increase scale")
        
        return should_increase
                 
    def update_on_small_grad(self) -> None:
        """
        Update the scaling factor when gradients are small.
        
        Multiplies the current scaling factor by increase_factor to scale up
        small gradients and improve precision utilization.
        """
        old_S = self.S
        self.S *= self.increase_factor
        if self.verbose:
            print(f"[DynamicScaler] Small gradient: scale increased from {old_S:.6e} to {self.S:.6e}")
        self.scale_history.append(('increase', self.S))
    