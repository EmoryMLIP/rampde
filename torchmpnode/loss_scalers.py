import torch
import math


class NoScaler:
    """
    A dummy class that does not perform any scaling.
    This is useful for debugging or when scaling is not needed.
    """
    def __init__(self, dtype_low):
        self.dtype_low = dtype_low
        self.is_initialized=True
        self.max_attempts = 1
        self.S = 1.0
        self.__name__="NoScaler"

    def init_scaling(self, a):
        # No scaling needed, so we do nothing.
        pass

    def _is_any_infinite(self, x):
        """
        Recursively check if x (a tensor, list, or tuple of tensors) contains any non-finite values.
        Returns True if any tensor element is inf or NaN; otherwise False.
        """
        if x is None:
            return False
        if isinstance(x, torch.Tensor):
            return not x.isfinite().all().item()
        if isinstance(x, (list, tuple)):
            return any(self._is_any_infinite(elem) for elem in x)
        # If x is neither, try to convert to tensor.
        try:
            return not torch.tensor(x).isfinite().all().item()
        except Exception:
            return False
        
    def check_for_increase(self,a):
        return False
    
    def update_on_overflow(self):
        # throw an error
        raise RuntimeError("Overflow detected, but NoScaler does not handle scaling.")
    
    def update_on_small_grad(self):
        pass


class DynamicScaler(NoScaler):
    def __init__(self, dtype_low, target_factor=None, increase_factor=2.0, decrease_factor=0.5,
                  max_attempts=10, delta=0):
        super().__init__(dtype_low)
        # Set a target norm if not provided: 1/epsilon for low precision.
        self.eps = torch.finfo(dtype_low).eps
        self.target = target_factor if target_factor is not None else 1.0 / self.eps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_attempts = max_attempts
        self.delta = delta
        self.is_initialized=False
        self.S = None  # This will be initialized later
        self.__name__=  "DynamicScaler"

    def init_scaling(self, a):
        if not(a.isfinite().all()) or a.isnan().any():
            raise ValueError("Input tensor contains non-finite or nan values.")
        
        # get the number of elements in a except for the 0th dimension
        target = self.target / math.sqrt(a.numel() / a.shape[0])
        self.S = target / (a.abs().max() + self.delta).to(torch.float32)
        self.S = 2**(torch.round(torch.log2(self.S))).item()
        # make sure S is a power of 2
        for _ in range(20):         # 20 halvings = divide by 1 048 576
            anew = self.S * a
            if anew.isfinite().all():
                break
            self.S *= 0.5
        else:
            raise RuntimeError(f"Scaler failed to find finite scale after 20 steps for {a.shape} with ||a||_inf = {a.abs().max()}.")

    def update_on_overflow(self):
        """
        Update the scaling factor on overflow (multiply S by decrease_factor) and
        scale each of the provided arguments by decrease_factor. Returns a tuple of
        scaled inputs in the same order. Optionally apply scaling in place.
        """
        self.S *= self.decrease_factor

    def check_for_increase(self, a):
        # Use .item() to return a Python bool, not a tensor
        return ((a.abs().max()) / self.target < 0.5).item()
                 
    def update_on_small_grad(self):
        """
        Update the scaling factor on small gradients (multiply S by increase_factor) and
        scale each of the provided arguments by increase_factor. Returns a tuple of
        scaled inputs in the same order. Optionally apply scaling in place.
        """
        self.S *= self.increase_factor
    