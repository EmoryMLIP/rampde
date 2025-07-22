"""
Main ODE integration interface with automatic solver selection.

This module provides the main odeint function that automatically selects
the optimal solver variant based on the loss scaler type and precision.
"""

from typing import Union, Optional, Tuple, Literal, TypeVar, Callable, Type, Any
import torch
from .increment import INCREMENTS, get_increment_function
from .fixed_grid_unscaled import FixedGridODESolverUnscaled
from .fixed_grid_dynamic import FixedGridODESolverDynamic
from .fixed_grid_unscaled_safe import FixedGridODESolverUnscaledSafe
from .loss_scalers import DynamicScaler

# Type definitions
TensorType = TypeVar('TensorType', bound=torch.Tensor)
TupleOrTensor = Union[torch.Tensor, Tuple[torch.Tensor, ...]]
ScalerType = Union[DynamicScaler, None, Literal[False]]
MethodType = Literal['euler', 'rk4']


def _tensor_to_tuple(
    tensor: torch.Tensor, 
    numels: list[int], 
    shapes: list[torch.Size], 
    length: Tuple[int, ...]
) -> Tuple[torch.Tensor, ...]:
    """Convert tensor to tuple of tensors."""
    tup = torch.split(tensor, numels, dim=-1)
    return tuple([t.view((*length, *s)) for t, s in zip(tup, shapes)])


def _tuple_to_tensor(tup: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Convert tuple of tensors to single tensor."""
    return torch.cat([t for t in tup], dim=-1)


class _TupleFunc(torch.nn.Module):
    """
    Wrapper for ODE functions that work with tuples.
    Taken from torchdiffeq.
    """
    def __init__(
        self, 
        base_func: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]], 
        shapes: list[torch.Size], 
        numels: list[int]
    ):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes
        self.numels = numels

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f = self.base_func(t, _tensor_to_tuple(y, self.numels, self.shapes, ()))
        return _tuple_to_tensor(f)


def _select_ode_solver(
    loss_scaler: ScalerType, 
    precision: torch.dtype
) -> Tuple[Type[torch.autograd.Function], Optional[DynamicScaler]]:
    """
    Select optimal ODE solver based on scaler type and precision.
    
    Creates default loss scaler if None, then selects appropriate solver variant.
    
    Args:
        loss_scaler: Loss scaler instance (DynamicScaler, None, or False).
                    - If None, uses automatic scaling based on precision:
                      * DynamicScaler for float16 under autocast
                      * None (no scaling) for other precisions
                    - If False, explicitly disables all internal scaling
                    - If DynamicScaler instance, uses that scaler
        precision: Precision dtype (torch.float32, torch.float16, torch.bfloat16)
        
    Returns:
        tuple: (ODE solver class, loss_scaler) optimized for the given configuration
    """
    # Handle explicit False to disable all internal scaling
    if loss_scaler is False:
        loss_scaler = None  # Convert False to None for solver selection
    # Create default loss scaler if not provided
    elif loss_scaler is None:
        dtype_low = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        if dtype_low == torch.float16:
            loss_scaler = DynamicScaler(dtype_low=dtype_low)
        # else: loss_scaler remains None
    
    if isinstance(loss_scaler, DynamicScaler):
        # Dynamic scaling required - use scaling loop variant
        return FixedGridODESolverDynamic, loss_scaler
    
    elif loss_scaler is None:
        if precision in [torch.float32, torch.bfloat16]:
            # Optimal performance for stable precisions
            return FixedGridODESolverUnscaled, loss_scaler
        else:
            # May need overflow protection for float16
            return FixedGridODESolverUnscaledSafe, loss_scaler
    
    else:
        # Default to safe version for unknown scalers
        return FixedGridODESolverUnscaledSafe, loss_scaler


def odeint(
    func: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]], 
    y0: TupleOrTensor, 
    t: torch.Tensor, 
    *, 
    method: MethodType = 'rk4', 
    atol: Optional[float] = None, 
    rtol: Optional[float] = None, 
    loss_scaler: ScalerType = None
) -> TupleOrTensor:
    """
    Solve an ODE system with automatic solver selection.
    
    This function automatically selects the optimal solver variant based on
    the loss scaler type and precision to provide the best performance for
    each configuration.
    
    Args:
        func: ODE function f(t, y)
        y0: Initial condition (tensor or tuple of tensors)
        t: Time points tensor
        method: Integration method ('rk4' or 'euler')
        atol: Absolute tolerance (unused in fixed grid solvers)
        rtol: Relative tolerance (unused in fixed grid solvers)
        loss_scaler: Loss scaler for mixed precision (DynamicScaler, None, or False).
                    - If None, creates default scaler:
                      * DynamicScaler for float16 under autocast
                      * None (no scaling) for other precisions
                    - If False, explicitly disables all internal scaling
                    - If DynamicScaler instance, uses that scaler
    
    Returns:
        Solution tensor or tuple of tensors
        
    Solver Selection Logic:
        - DynamicScaler: Uses FixedGridODESolverDynamic (scaling loop)
        - None + float32/bfloat16: Uses FixedGridODESolverUnscaled (optimal)
        - None + float16: Uses FixedGridODESolverUnscaledSafe (exception handling)
        - Other scalers: Uses FixedGridODESolverUnscaledSafe (safe default)
    """
    # Handle tuple inputs
    y0_tuple = isinstance(y0, tuple)
    if y0_tuple:
        shapes = [y0_i.shape for y0_i in y0]
        numels = [s[-1] for s in shapes]

        # Choose tuple wrapper based on optimization setting
        func = _TupleFunc(func, shapes, numels)
        y0 = _tuple_to_tensor(y0)
    
    # Get increment function
    increment_func = get_increment_function(method)()
    
    # Determine precision for solver selection
    precision = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
    
    # Automatically select optimal solver variant and create loss scaler if needed
    solver_class, loss_scaler = _select_ode_solver(loss_scaler, precision)
    
    # Collect ODE function parameters
    params = func.parameters()
    
    # Solve the ODE using the selected solver
    solution = solver_class.apply(increment_func, func, y0, t, loss_scaler, *params)
    
    # Convert back to tuple if needed
    if y0_tuple:
        return _tensor_to_tuple(solution, numels, shapes, (len(t),))
    else:
        return solution