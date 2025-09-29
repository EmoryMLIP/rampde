# Phi.py -- Transpose-free mixed precision optimized neural network for potential function
# Optimized version with tensor core support and numerical stability improvements
import torch
import torch.nn as nn
import copy
from torch.amp import custom_fwd

@torch.jit.script
def combined_tanh_antideriv(x, cast: bool = True):
    """
    Combined computation of antiderivative and tanh with single cast operation.
    
    Returns:
        tuple: (antideriv_tanh, tanh_values) where:
        - antideriv_tanh: int tanh dx = |x| + log(1+exp(-2|x|))
        - tanh_values: tanh(x)
    
    This optimization reduces casting overhead and enables reuse of tanh values
    for derivative computation (d/dx tanh = 1 - tanh²(x)).
    """
    dtype = x.dtype  # Initialize dtype outside the if block for JIT compatibility
    
    if cast:
        x = x.to(torch.float32)
    
    # Compute tanh once and reuse it
    tanh_x = torch.tanh(x)
    
    # Compute antiderivative: |x| + log(1+exp(-2|x|))
    antideriv = torch.abs(x) + torch.log1p(torch.exp(-2.0 * torch.abs(x)))
    
    if cast:
        return antideriv.to(dtype), tanh_x.to(dtype)
    else:
        return antideriv, tanh_x

def antiderivTanh(x, cast=True):
    """
    int tanh dx = |x| + log(1+exp(-2|x|))
    use log1p for numerical stability, see tests/test_act.py
    If cast=True, keep the computation in f32, only cast to low precision in the output
    """
    if cast:
        dtype = x.dtype
        x = x.to(torch.float32)
    
    act = torch.abs(x) + torch.log1p(torch.exp(-2.0 * torch.abs(x)))
    return act.to(dtype) if cast else act
    




@torch.jit.script
def jit_backward_step(tanh_output, term_T, weight_i, h: float):
    """
    JIT-compiled backward pass computation for a single ResNet layer.
    
    Args:
        tanh_output: (nex, m) - precomputed tanh values
        term_T: (1, m) or (nex, m) - term for backpropagation
        weight_i: (m, m) - layer weights
        h: float - step size
    
    Returns:
        z_T: (nex, m) - updated backprop values
    """
    # Element-wise multiplication with broadcasting
    element_wise = tanh_output * term_T  # (nex, m) * (1, m) -> (nex, m)
    
    # Matrix multiplication
    dz_T = h * torch.mm(element_wise, weight_i)  # (nex, m) @ (m, m) -> (nex, m)
    
    return term_T + dz_T

@torch.jit.script
def jit_trace_step(trH, weight_i, Jac, tanh_val, term_val, h: float, skipJacobian: bool = False):
    """
    JIT-compiled trace computation for a single ResNet layer.
    
    Args:
        weight_i: (m, m) - layer weights
        Jac: (nex, m, d+1) - Jacobian matrix
        tanh_val: (nex, m) - precomputed tanh values
        term_val: (1, m) or (nex, m) - term for trace computation
        h: float - step size
    
    Returns:
        t_i: (nex,) - trace contribution
        Jac_updated: (nex, m, d+1) - updated Jacobian
    """
    # Apply weight transformation to spatial Jacobian
    KJ = torch.einsum('ij,bjk->bik', weight_i, Jac)
    
    # Compute derivative term
    deriv_term = (1.0 - tanh_val * tanh_val) * term_val  # (nex, m) * (nex, m) -> (nex, m)
    
    # Compute trace contribution
    KJ_squared = torch.norm(KJ, dim=2)**2  # (nex, m, d) -> (nex, m)
    t_i = torch.sum(deriv_term * KJ_squared, dim=1)  # (nex, m) * (nex, m) -> (nex,)
    trH = trH + h * t_i
    # Update Jacobian
    if not skipJacobian:
        tanh_temp = tanh_val.unsqueeze(2)  # (nex, m, 1)
        Jac = Jac + h * tanh_temp * KJ  # Update spatial part only
    
    return trH, Jac

@torch.jit.script
def jit_element_wise_ops(tanh_output, term_T, weight_0):
    """
    JIT-compiled element-wise operations for z_0 computation.
    
    Args:
        tanh_output: (nex, m) - precomputed tanh values
        term_T: (nex, m) - term for backpropagation
        weight_0: (m, d+1) - first layer weights
    
    Returns:
        z_0: (nex, d+1) - final backprop values
    """
    element_wise_0 = tanh_output * term_T  # (nex, m) * (nex, m) -> (nex, m)
    return torch.mm(element_wise_0, weight_0)  # (nex, m) @ (m, d+1) -> (nex, d+1)

@torch.jit.script
def jit_opening_trace_step(weight_0, opening_tanh, z_T_1, tanhopen):
    """
    JIT-compiled opening layer trace and Jacobian computation.
    
    Args:
        weight_0: (m, d+1) - opening layer weights
        opening_tanh: (nex, m) - precomputed tanh values for opening layer
        z_T_1: (nex, m) - backprop values from layer 1
        tanhopen: (nex, m) - tanh values for Jacobian computation
    
    Returns:
        tuple: (trH, Jac) where:
        - trH: (nex,) - trace contribution from opening layer
        - Jac: (nex, m, d+1) - Jacobian matrix
    """
    # Create Kopen with last column zeroed for mathematical equivalence
    Kopen = weight_0.clone()      # (m, d+1)
    Kopen[:, -1] = 0              # Zero last column
    
    # Optimized trace computation using precomputed tanh values
    temp = (1.0 - opening_tanh * opening_tanh) * z_T_1  # derivTanh_from_tanh(opening_tanh) * z_T[1]
    Kopen_norm_sq = torch.norm(Kopen, dim=1)**2  # (m, d+1) -> (m,) ||Kopen||^2 per row
    trH = temp @ Kopen_norm_sq              # (nex, m) @ (m,) -> (nex,)
    
    # Compute Jacobian directly with natural shape
    Jac = tanhopen.unsqueeze(2) * Kopen.unsqueeze(0)  # (nex, m, 1) * (1, m, d+1) -> (nex, m, d+1)
    
    return trH, Jac

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi with mixed precision optimization
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.combined_act = combined_tanh_antideriv  # JIT optimized combined function
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x):
        """
            Forward pass of the ResNet with mixed precision optimization
        """
        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x

class Phi(nn.Module):
    def __init__(self, nTh, m, d, r=10, alph=[1.0] * 5):
        """
            Transpose-free mixed precision optimized neural network approximating Phi
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d
        self.alph = alph

        r = min(r,d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d, m, nTh=nTh)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        if self.c.bias is not None:
            self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x):
        """ calculating Phi(s, theta) with mixed precision optimization """

        # force A to be symmetric - optimize matrix multiplication for tensor cores
        A_t = self.A.t().contiguous()
        symA = torch.mm(A_t, self.A) # A'A

        # Optimize the quadratic form computation
        x_symA = torch.mm(x, symA)
        quadratic_term = 0.5 * torch.sum(x_symA * x, dim=1, keepdim=True)
        
        return self.w(self.N(x)) + quadratic_term + self.c(x)

    @custom_fwd(cast_inputs=torch.float16)
    def trHess(self, x, justGrad=False, print_prec=False):
        """
        Transpose-free mixed precision optimized computation of gradient and trace(Hessian)
        
        Key innovation: Compute z^T throughout to eliminate transpose operations
        """

        # Get autocast dtype for final conversion
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32

        N = self.N
        d = x.shape[1] - 1
        
        # Optimize symmetric matrix computation
        A_t = self.A.t().contiguous()
        symA = torch.mm(A_t, self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z_T = N.nTh * [None] # hold the z_0^T,z_1^T,...,z_M^T (TRANSPOSED!) for the backward pass
        tanh_values = [] # cache tanh values for efficient reuse in backward pass
        antideriv_values = [] # cache antiderivative values

        # Forward of ResNet N and fill u (optimized with combined function)
        opening = N.layers[0].forward(x) # K_0 * S + b_0
        if print_prec:
            print("opening dtype", opening.dtype, "opening device", opening.device)
        
        # Use combined function for opening layer
        opening_antideriv, opening_tanh = N.combined_act(opening)
        u.append(opening_antideriv) # u0
        feat = u[0]

        for i in range(1, N.nTh):
            layer_output = N.layers[i](feat)  # Raw layer computation
            # Use combined function to get both antiderivative and tanh values
            antideriv, tanh_val = N.combined_act(layer_output)
            
            # Store values for backward pass
            tanh_values.append(tanh_val)
            antideriv_values.append(antideriv)
            
            df = N.h * antideriv
            if print_prec:
                print("df dtype", layer_output.dtype, "df device", layer_output.device)
            feat = feat + df
            u.append(feat)

        # Use precomputed tanh value from combined function
        tanhopen = opening_tanh # act'( K_0 * S + b_0 ) - already computed

        # TRANSPOSE-FREE GRADIENT COMPUTATION
        # Compute z^T instead of z to eliminate transposes
        for i in range(N.nTh-1, 0, -1): # work backwards, placing z_i^T in appropriate spot
            if i == N.nTh-1:
                term_T = self.w.weight  # (1, m)
            else:
                term_T = z_T[i+1]  # Already transposed from previous iteration

            # JIT-OPTIMIZED BACKWARD PASS
            z_T[i] = jit_backward_step(tanh_values[i-1], term_T, N.layers[i].weight, N.h)
            
            if print_prec:
                print("z_T[i] dtype", z_T[i].dtype, "z_T[i] device", z_T[i].device)

        # z_0^T computation: z_0^T = (tanhopen * z_1^T) @ W_0
        z_T[0] = jit_element_wise_ops(tanhopen, z_T[1],  N.layers[0].weight)
        
        if print_prec:
            print("z_T[0] dtype", z_T[0].dtype, "z_T[0] device", z_T[0].device)
        
        grad_T = z_T[0] + torch.mm(x, symA) + self.c.weight  # (nex, d+1) + (nex, d+1) + (1, d+1) -> (nex, d+1)
        
        if justGrad:
            # Return gradient in transposed form (which is the natural form now)
            return grad_T.to(dtype_low)

        # -----------------
        # trace of Hessian (updated for consistency with transposed z values)
        #-----------------

        # JIT-compiled opening layer trace and Jacobian computation
        trH, Jac = jit_opening_trace_step(N.layers[0].weight, opening_tanh, z_T[1], tanhopen)

        for i in range(1, N.nTh):
            if i == N.nTh-1:
                term_val = self.w.weight                # (1, m) - final layer weights
            else:
                term_val = z_T[i+1]                    # (nex, m) - backprop values

            # JIT-compiled trace computation for optimal performance
            tanh_val = tanh_values[i-1]  # Use cached tanh value (nex, m)
            trH, Jac = jit_trace_step(trH,N.layers[i].weight , Jac, tanh_val, term_val, N.h, i< N.nTh-1)

        final_trace = trH + torch.trace(symA[0:d,0:d])  # (nex,) + scalar -> (nex,)
        
        return grad_T.to(dtype_low), final_trace.to(dtype_low)


if __name__ == "__main__":

    import time