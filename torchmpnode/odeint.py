import torch
from .fixed_grid import Euler, RK4, FixedGridODESolver

SOLVERS = {'euler': Euler, 'rk4': RK4}

def _tensor_to_tuple(tensor,numels,shapes,length):
    tup = torch.split(tensor, numels, dim=-1)
    return tuple([t.view((*length,*s)) for t,s in zip(tup,shapes)])
    
def _tuple_to_tensor(tup):
    return torch.cat([t for t in tup],dim=-1)

class _TupleFunc(torch.nn.Module):
    """
    Taken from torchdiffeq
    """
    def __init__(self, base_func, shapes, numels):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes
        self.numels = numels

    def forward(self, t, y):
        f =  self.base_func(t, _tensor_to_tuple(y,self.numels,self.shapes,()))
        return _tuple_to_tensor(f)

def odeint(func, y0, t, *, method='rk4', atol=None, rtol=None):
    y0_tuple = isinstance(y0, tuple)
    if y0_tuple:
        shapes = [y0_i.shape for y0_i in y0]
        numels = [s[-1] for s in shapes]

        func = _TupleFunc(func, shapes, numels)
        y0 = _tuple_to_tensor(y0)
        
    solver = SOLVERS[method]()
    params = func.parameters()
    solution =  FixedGridODESolver.apply(solver,func, y0, t, *params)
    if y0_tuple:
        return _tensor_to_tuple(solution,numels,shapes,(len(t),))
    else:
        return solution


