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
        self.__name__="NoScaler"

    def init_scaling(self, a):
        # No scaling needed, so we do nothing.
        pass

    def scale(self, tensor, in_place=False):
        # No scaling needed, so we return the tensor as is.
        return tensor

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
    
    def unscale(self, tensor, in_place=False):
        return tensor

    def update_on_overflow(self, *args, in_place=False):
        # throw an error
        raise RuntimeError("Overflow detected, but NoScaler does not handle scaling.")
        return args

    def update_on_small_grad(self, *args, in_place=False):
        return args


class DynamicScaler(NoScaler):
    def __init__(self, dtype_low, target_factor=None, increase_factor=2.0, decrease_factor=0.5,
                 small_grad_threshold=1.0, max_attempts=10, delta=0):
        super().__init__(dtype_low)
        # Set a target norm if not provided: 1/epsilon for low precision.
        self.eps = torch.finfo(dtype_low).eps
        self.target = target_factor if target_factor is not None else 1.0 / self.eps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.small_grad_threshold = small_grad_threshold
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

            
    
    def _apply_scaling(self, x, factor, in_place=False):
        """
        Multiply x by factor. If in_place is True and x is a tensor, modify it in-place.
        Supports x as a tensor, or list/tuple of tensors.
        """
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            scaled = []
            for elem in x:
                scaled.append(self._apply_scaling(elem, factor, in_place=in_place))
            return type(x)(scaled)
        else:
            if in_place:
                # Use in-place multiplication if possible
                x.mul_(factor)
                return x
            else:
                return x * factor

    def scale(self, tensor, in_place=False):
        """Scale the tensor (or collection of tensors) by S. Optionally in-place."""
        return self._apply_scaling(tensor, self.S, in_place=in_place)

    def unscale(self, tensor, in_place=False):
        """Unscale the tensor (or collection of tensors) by dividing by S. Optionally in-place."""
        return self._apply_scaling(tensor, 1.0 / self.S, in_place=in_place)

    def update_on_overflow(self, *args, in_place=False):
        """
        Update the scaling factor on overflow (multiply S by decrease_factor) and
        scale each of the provided arguments by decrease_factor. Returns a tuple of
        scaled inputs in the same order. Optionally apply scaling in place.
        """
        self.S *= self.decrease_factor
        scaled_args = []
        for arg in args:
            scaled_args.append(self._apply_scaling(arg, self.decrease_factor, in_place=in_place))
        return tuple(scaled_args)

    def check_for_increase(self, a):
        # Use .item() to return a Python bool, not a tensor
        return ((a.abs().max()) / self.target < 0.5).item()
                 
    def update_on_small_grad(self, *args, in_place=False):
        """
        Update the scaling factor on small gradients (multiply S by increase_factor) and
        scale each of the provided arguments by increase_factor. Returns a tuple of
        scaled inputs in the same order. Optionally apply scaling in place.
        """
        self.S *= self.increase_factor
        scaled_args = []
        for arg in args:
            scaled_args.append(self._apply_scaling(arg, self.increase_factor, in_place=in_place))
        return tuple(scaled_args)