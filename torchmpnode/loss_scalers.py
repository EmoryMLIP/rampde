import torch




class DynamicScaler:
    def __init__(self, dtype_low, target_factor=None, increase_factor=2.0, decrease_factor=0.5,
                 small_grad_threshold=1.0, max_attempts=10, delta=0):
        self.dtype_low = dtype_low
        # Set a target norm if not provided: 1/epsilon for low precision.
        self.eps = torch.finfo(dtype_low).eps
        self.target = target_factor if target_factor is not None else 1.0 / self.eps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.small_grad_threshold = small_grad_threshold
        self.max_attempts = max_attempts
        self.delta = delta
        self.S = None  # This will be initialized later

    def init_scaling(self, a):
        # Initialize S such that S * ||a|| ~ target.
        self.S = self.target / (a.norm() + self.delta).to(torch.float32)
        self.S = 2**(torch.round(torch.log2(self.S))).item()

        # make sure S is a power of 2
        anew = self.S * a
        while not(anew.isfinite().all()):
            print('inf')
            self.S *= 0.5
            anew = self.S * a
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