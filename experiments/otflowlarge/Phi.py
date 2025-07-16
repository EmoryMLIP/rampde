# Phi.py -- Transpose-free mixed precision optimized neural network for potential function
# Optimized version with tensor core support and numerical stability improvements
import torch
import torch.nn as nn
import copy
from torch.amp import custom_fwd

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
    
def derivTanh(x, cast=False):
    """
    d/dx tanh = 1 - tanh(x)^2
    If cast=True, keep the computation in f32, only cast to low precision in the output
    """
    if cast:
        dtype = x.dtype
        x = x.to(torch.float32)
    act = 1.0 - torch.tanh(x).pow(2)
    return act.to(dtype) if cast else act

def optimize_for_tensor_cores(tensor):
    """
    Optimize tensor shape for tensor core utilization.
    """
    return tensor.contiguous()

def efficient_matmul(a, b):
    """
    Efficient matrix multiplication that tries to use tensor cores when possible.
    """
    # Ensure tensors are contiguous for better performance
    a = a.contiguous()
    b = b.contiguous()
    
    # Use torch.mm for 2D tensors as it's optimized for tensor cores
    if a.dim() == 2 and b.dim() == 2:
        return torch.mm(a, b)
    else:
        return torch.matmul(a, b)

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
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    @custom_fwd(device_type='cuda')
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

    @custom_fwd(device_type='cuda')
    def forward(self, x):
        """ calculating Phi(s, theta) with mixed precision optimization """

        # force A to be symmetric - optimize matrix multiplication for tensor cores
        A_t = optimize_for_tensor_cores(self.A.t())
        symA = efficient_matmul(A_t, self.A) # A'A

        # Optimize the quadratic form computation
        x_opt = optimize_for_tensor_cores(x)
        x_symA = efficient_matmul(x_opt, symA)
        quadratic_term = 0.5 * torch.sum(x_symA * x_opt, dim=1, keepdim=True)
        
        return self.w(self.N(x)) + quadratic_term + self.c(x)

    @custom_fwd(device_type='cuda')
    def trHess(self, x, justGrad=False, print_prec=False):
        """
        Transpose-free mixed precision optimized computation of gradient and trace(Hessian)
        
        Key innovation: Compute z^T throughout to eliminate transpose operations
        """

        # Get autocast dtype for final conversion
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32

        N = self.N
        m = N.layers[0].weight.shape[0]
        nex = x.shape[0] # number of examples in the batch
        d = x.shape[1] - 1
        
        # Optimize symmetric matrix computation
        A_t = optimize_for_tensor_cores(self.A.t())
        symA = efficient_matmul(A_t, self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z_T = N.nTh * [None] # hold the z_0^T,z_1^T,...,z_M^T (TRANSPOSED!) for the backward pass
        layer_outputs = [] # cache raw layer outputs to avoid recomputation

        # Forward of ResNet N and fill u (same as before)
        opening = N.layers[0].forward(x) # K_0 * S + b_0
        if print_prec:
            print("opening dtype", opening.dtype, "opening device", opening.device)
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1, N.nTh):
            layer_output = N.layers[i](feat)  # Cache this computation
            layer_outputs.append(layer_output)  # Store for reuse
            df = N.h * N.act(layer_output)
            if print_prec:
                print("df dtype", layer_output.dtype, "df device", layer_output.device)
            feat = feat + df
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # TRANSPOSE-FREE GRADIENT COMPUTATION
        # Compute z^T instead of z to eliminate transposes
        for i in range(N.nTh-1, 0, -1): # work backwards, placing z_i^T in appropriate spot
            if i == N.nTh-1:
                # Instead of: term = self.w.weight.t()  # (m, 1) -> (1, m)
                # Use: term_T = self.w.weight  # Keep as (1, m)
                term_T = self.w.weight  # (1, m)
            else:
                term_T = z_T[i+1]  # Already transposed from previous iteration

            # CORE OPTIMIZATION: Eliminate transposes
            # Old: z_i = W_i^T * (tanh_output^T * term)
            # New: z_i^T = (tanh_output * term_T) @ W_i
            layer_output = layer_outputs[i-1]  # Use cached output (nex, m)
            tanh_output = torch.tanh(layer_output)       # (nex, m)
            
            # Reshape term_T to be broadcastable with tanh_output
            # term_T: (1, m) -> (nex, m) via broadcasting
            element_wise = tanh_output * term_T  # (nex, m) * (1, m) -> (nex, m)
            
            # Matrix multiply to get z_i^T: (nex, m) @ (m, m) -> (nex, m)
            weight_i = optimize_for_tensor_cores(N.layers[i].weight)  # (m, m)
            dz_T = N.h * efficient_matmul(element_wise, weight_i)  # (nex, m)
            
            if print_prec:
                print("dz_T dtype", dz_T.dtype, "dz_T device", dz_T.device)
            
            z_T[i] = term_T + dz_T  # (1, m) + (nex, m) -> (nex, m) via broadcasting

        # z_0^T computation: z_0^T = (tanhopen * z_1^T) @ W_0
        # tanhopen: (nex, m), z_T[1]: (nex, m), W_0: (m, d+1)
        element_wise_0 = tanhopen * z_T[1]  # (nex, m) * (nex, m) -> (nex, m)
        weight_0 = optimize_for_tensor_cores(N.layers[0].weight)  # (m, d+1)
        z_T[0] = efficient_matmul(element_wise_0, weight_0)  # (nex, m) @ (m, d+1) -> (nex, d+1)
        
        if print_prec:
            print("z_T[0] dtype", z_T[0].dtype, "z_T[0] device", z_T[0].device)
        
        # Final gradient computation (transpose-free)
        # Old: grad = z[0] + symA @ x^T + c.weight^T
        # New: grad^T = z[0]^T + x @ symA^T + c.weight
        # symA_t = optimize_for_tensor_cores(symA)
        x_symA_t = efficient_matmul(x, symA)  # (nex, d+1) @ (d+1, d+1) -> (nex, d+1)
        grad_T = z_T[0] + x_symA_t + self.c.weight  # (nex, d+1) + (nex, d+1) + (1, d+1) -> (nex, d+1)
        
        if justGrad:
            # Return gradient in transposed form (which is the natural form now)
            return grad_T.to(dtype_low)

        # -----------------
        # trace of Hessian (updated for consistency with transposed z values)
        #-----------------

        # t_0, the trace of the opening layer - TENSOR CORE OPTIMIZED
        # Mathematical insight: E^T Hess E where E is (d+1)×d identity-like matrix
        # Equivalent to using (d+1)×(d+1) matrix with last column zeroed
        Kopen = N.layers[0].weight.clone()      # (m, d+1) - tensor core friendly
        Kopen[:, -1] = 0                        # Zero last column for mathematical equivalence
        
        # Optimized trace computation using tensor cores
        temp = derivTanh(opening) * z_T[1]      # (nex, m) * (nex, m) -> (nex, m)
        Kopen_norm_sq = torch.norm(Kopen, dim=1)**2  # (m, d+1) -> (m,) ||Kopen||^2 per row - numerically stable
        trH = temp @ Kopen_norm_sq              # (nex, m) @ (m,) -> (nex,) - tensor core efficient

        # grad_s u_0 ^ T - Jacobian computation with shape annotations  
        temp = tanhopen                             # (nex, m) - no transpose needed
        # Compute Jacobian directly with natural shape
        Jac = temp.unsqueeze(2) * Kopen.unsqueeze(0)  # (nex, m, 1) * (1, m, d+1) -> (nex, m, d+1)

        # t_i, trace of the ResNet layers - optimized for transpose-free computation
        for i in range(1, N.nTh):
            # Apply weight transformation to spatial Jacobian (nex, m, d)
            KJ = torch.bmm(weight_i.expand(nex,m,m),Jac)
            
            if i == N.nTh-1:
                term_val = self.w.weight                # (1, m) - final layer weights
            else:
                term_val = z_T[i+1]                    # (nex, m) - backprop values

            layer_output = layer_outputs[i-1]  # Use cached output (nex, m)
            # No transpose needed - keep natural (nex, m) shape
            # Handle broadcasting for different term_val shapes  
            deriv_term = derivTanh(layer_output) * term_val      # (nex, m) * (nex, m) -> (nex, m)
            
            # Expand to match KJ dimensions for trace computation
            KJ_squared = torch.norm(KJ, dim=2)**2  # (nex, m, d) -> (nex, m) - numerically stable squared norms
            t_i = torch.sum(deriv_term * KJ_squared, dim=1)  # (nex, m) * (nex, m) -> (nex,)
            
            trH = trH + N.h * t_i                      # (nex,) accumulate trace
            
            # Update Jacobian: add contribution to spatial dimensions only
            tanh_temp = torch.tanh(layer_output).unsqueeze(2)  # (nex, m, 1)
            Jac = Jac + N.h * tanh_temp * KJ  # Update spatial part only

        # Return transposed gradient and trace
        # Final trace: add quadratic form contribution
        # symA[0:d,0:d] is the spatial part of the symmetric matrix
        final_trace = trH + torch.trace(symA[0:d,0:d])  # (nex,) + scalar -> (nex,)
        
        return grad_T.to(dtype_low), final_trace.to(dtype_low)


if __name__ == "__main__":

    import time