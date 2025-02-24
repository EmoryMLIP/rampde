import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from torch.amp import autocast, GradScaler
#print adjoint values dtype in augmented_dynamics in adjoint.py

# >> Inside odeint_adjoint
# >> Adjoint state dtype at t=1.0: torch.float32
# >> Adjoint state dtype at t=0.9629629850387573: torch.float32
# >> Adjoint state dtype at t=0.9259259104728699: torch.float32
# >> Adjoint state dtype at t=0.8888888955116272: torch.float32
# >> Adjoint state dtype at t=0.8888888955116272: torch.float32
# >> Adjoint state dtype at t=0.8518518805503845: torch.float32
# >> Adjoint state dtype at t=0.8148148059844971: torch.float32
# >> Adjoint state dtype at t=0.7777777910232544: torch.float32
# >> Adjoint state dtype at t=0.7777777910232544: torch.float32
# >> Adjoint state dtype at t=0.7407407760620117: torch.float32
# >> Adjoint state dtype at t=0.7037037014961243: torch.float32
# >> Adjoint state dtype at t=0.6666666865348816: torch.float32
# >> Adjoint state dtype at t=0.6666666865348816: torch.float32
# >> Adjoint state dtype at t=0.6296296715736389: torch.float32
# >> Adjoint state dtype at t=0.5925925970077515: torch.float32
# >> Adjoint state dtype at t=0.5555555820465088: torch.float32
# >> Adjoint state dtype at t=0.5555555820465088: torch.float32
# >> Adjoint state dtype at t=0.5185185670852661: torch.float32
# >> Adjoint state dtype at t=0.48148149251937866: torch.float32
# >> Adjoint state dtype at t=0.4444444477558136: torch.float32
# >> Adjoint state dtype at t=0.4444444477558136: torch.float32
# >> Adjoint state dtype at t=0.40740740299224854: torch.float32
# >> Adjoint state dtype at t=0.37037038803100586: torch.float32
# >> Adjoint state dtype at t=0.3333333432674408: torch.float32
# >> Adjoint state dtype at t=0.3333333432674408: torch.float32
# >> Adjoint state dtype at t=0.29629629850387573: torch.float32
# >> Adjoint state dtype at t=0.25925925374031067: torch.float32
# >> Adjoint state dtype at t=0.2222222238779068: torch.float32
# >> Adjoint state dtype at t=0.2222222238779068: torch.float32
# >> Adjoint state dtype at t=0.18518519401550293: torch.float32
# >> Adjoint state dtype at t=0.14814814925193787: torch.float32
# >> Adjoint state dtype at t=0.1111111119389534: torch.float32
# >> Adjoint state dtype at t=0.1111111119389534: torch.float32
# >> Adjoint state dtype at t=0.07407407462596893: torch.float32
# >> Adjoint state dtype at t=0.03703703731298447: torch.float32
# >> Adjoint state dtype at t=0.0: torch.float32
##################



class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)  

    def forward(self, t, x):
        out = self.linear(x)
        print(f"[Forward] Output dtype: {out.dtype}")
        return out
    

def test_adjoint_precision(use_autocast, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = ODEFunc().to(device)
    true_x0 = torch.randn(1, 2, device=device, requires_grad=True)
    t_span = torch.linspace(0, 1, 10, device=device)
    
    scaler = GradScaler()

    if use_autocast:
        with autocast(device_type='cuda', dtype=torch.float16):
            out = odeint(model, true_x0, t_span, method='rk4')
    else:
        out = odeint(model, true_x0, t_span, method='rk4')
    loss = out[-1].sum()
    scaler.scale(loss).backward()

test_adjoint_precision(use_autocast=True, seed=0)
test_adjoint_precision(use_autocast=False, seed=0)



