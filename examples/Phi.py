# Phi.py -- copied from https://github.com/EmoryMLIP/OT-Flow/blob/master/src/Phi.py
# neural network to model the potential function
import torch
import torch.nn as nn
import copy
from torch.amp import custom_fwd


# def antiderivTanh(x): # activation function aka the antiderivative of tanh
#     # print("antiderivTanh dtype", x.dtype, "antiderivTanh device", x.device)
#     return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

# def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
#     return 1 - torch.pow( torch.tanh(x) , 2 )


# def antiderivTanh(x, cast=True):
#     if cast:
#         dtype = x.dtype
#         x = x.to(torch.float16)
#     act =  torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))
#     if cast:
#         act = act.to(dtype)
#     return act

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
    
class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
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

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x



class Phi(nn.Module):
    def __init__(self, nTh, m, d, r=10, alph=[1.0] * 5):
        """
            neural network approximating Phi (see Eq. (9) in our paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
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



    def forward(self, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A

        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)


    def trHess(self,x, justGrad=False, print_prec=False ):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32

        N    = self.N
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]-1
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        if print_prec:
            print("opening dtype", opening.dtype, "opening device", opening.device)
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            df = N.h * N.act(N.layers[i](feat))
            if print_prec:
                print("df dtype", N.layers[i](feat).dtype, "df device", N.layers[i](feat).device)
            feat = feat + df
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            dz = N.h * torch.mm( N.layers[i].weight.t() , torch.tanh(N.layers[i].forward(u[i-1])).t() * term )
            if print_prec:
                print("dz dtype", dz.dtype, "dz device", dz.device)
            z[i] = term + dz
            # temp = N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)
            # print("temp dtype", temp.dtype, "z[i] device", z[i].device)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        if print_prec:
            print("z0 dtype", z[0].dtype, "z0 device", z[0].device)
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()
        # print("grad dtype", grad.dtype, "grad device", grad.device)
        
        if justGrad:
            return grad.t().to(dtype_low)

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = derivTanh(opening.t()) * z[1]
        
        
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1)) # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )

        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1,N.nTh):
            # print("trH dtype", trH.dtype, "trH device", trH.device)
        
            KJ  = torch.mm(N.layers[i].weight , Jac.reshape(m,-1) )
            if print_prec:
                print("KJ dtype", KJ.dtype, "KJ device", KJ.device)
            KJ  = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            if print_prec:
                print("temp dtype", temp.dtype, "temp device", temp.device)
            t_i = torch.sum(  ( derivTanh(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=(0, 1) )
            
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian

        return grad.t().to(dtype_low),( trH + torch.trace(symA[0:d,0:d])).to(dtype_low)
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )



if __name__ == "__main__":

    import time
    import math

    # test case
    d = 2
    m = 5

    net = Phi(nTh=2, m=m, d=d)
    net.N.layers[0].weight.data  = 0.1 + 0.0 * net.N.layers[0].weight.data
    net.N.layers[0].bias.data    = 0.2 + 0.0 * net.N.layers[0].bias.data
    net.N.layers[1].weight.data  = 0.3 + 0.0 * net.N.layers[1].weight.data
    net.N.layers[1].weight.data  = 0.3 + 0.0 * net.N.layers[1].weight.data

    # number of samples-by-(d+1)
    x = torch.Tensor([[1.0 ,4.0 , 0.5],[2.0,5.0,0.6],[3.0,6.0,0.7],[0.0,0.0,0.0]])
    y = net(x)
    print(y)

    # test timings
    d = 400
    m = 32
    nex = 1000

    net = Phi(nTh=5, m=m, d=d)
    net.eval()
    x = torch.randn(nex,d+1)
    y = net(x)

    end = time.time()
    g,h = net.trHess(x)
    print('traceHess takes ', time.time()-end)

    end = time.time()
    g = net.trHess(x, justGrad=True)
    print('JustGrad takes  ', time.time()-end)




