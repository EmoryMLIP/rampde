import torch
from .fixed_grid import Euler, RK4, FixedGridODESolver

SOLVERS = {'euler': Euler, 'rk4': RK4}

def odeint(func, y0, t, *, method='rk4', dtype_hi=torch.float32):

    solver = SOLVERS[method]()
    params = func.parameters()
    return FixedGridODESolver.apply(solver,func, y0, t, *params)


