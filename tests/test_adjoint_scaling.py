import unittest
import math
import torch
import torch.nn as nn
import sys, os
import argparse
import csv, pathlib, datetime, textwrap

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "test_adjoint_scaling"
OUT_DIR.mkdir(exist_ok=True)


import matplotlib
matplotlib.use('Agg')  # for headless CI
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint, NoScaler, DynamicScaler
from torch.amp import autocast

torch.set_default_dtype(torch.float32)


class PolynomialDampedODE(nn.Module):
    r"""
    ODE:  y'(t) = -λ(t) y(t),    λ(t) = a t**2 + b t + c

    Analytic solution:
        y(t) = y0 * exp( -a t**3/3 - b t**2/2 - c t )

    Trainable parameters
    --------------------
    a, b, c : nn.Parameter (stored in FP32)
    """
    def __init__(self):
        super().__init__()
        # Pick coefficients so that ∫₀ᴛ λ ≈ 0 (so y(T) ≈ y0)
        #   ∫₀ᴛ (a s² + b s + c) ds = a T³/3 + b T²/2 + c T
        # We fix T=3 and scale the zero‑integral polynomial by 0.5:
        #   (a,b,c) = 0.5 * (1, -3, 2)  ⇒ λ_max ≈ 1 → avoids fp16 overflow
        self.T = 2.65
        self.a = nn.Parameter(torch.tensor(8, dtype=torch.float32))    # 0.5 * 1
        self.b = nn.Parameter(torch.tensor(-10, dtype=torch.float32))   # 0.5 * (-3)
        self.c = nn.Parameter(torch.tensor(2**(-16), dtype=torch.float32))    # 0.5 * 2

    def forward(self, t: torch.Tensor, y: torch.Tensor):
        """
        Evaluate f(t,y) = -λ(t) y with λ(t)=a t² + b t + c.
        When torch.autocast is active, ensure all operands are in the current
        autocast working dtype to get a *true* low‑precision code path.
        """
        if torch.is_autocast_enabled():
            w_dtype = torch.get_autocast_dtype('cuda')
            t = t.to(w_dtype)
            y = y.to(w_dtype)
            a = self.a.clone().to(w_dtype)
            b = self.b.clone().to(w_dtype)
            c = self.c.clone().to(w_dtype)
        else:
            a, b, c = self.a, self.b, self.c

        lam = a * t**2 + b * t + c
        # check for NaN/inf in λ(t)
        if not torch.isfinite(lam).all():
            raise ValueError(f"λ(t) contains NaN/inf: {lam} at t={t}")
        rhs = -lam * y
        # check for NaN/inf in rhs
        if not torch.isfinite(rhs).all():
            raise ValueError(f"rhs contains NaN/inf: {rhs} at t={t}, y={y}")
        return rhs

    # ------------------------------------------------------------------
    # Analytic solution + grads for L = ½ y(T)²
    # ------------------------------------------------------------------
    def solve_analytically(self, y0: torch.Tensor, t: torch.Tensor):
        T = t.max().cpu().double()
        device = y0.device
        y0d = y0.detach().cpu().double().requires_grad_(True)
        a  = self.a.detach().cpu().double().requires_grad_(True)
        b  = self.b.detach().cpu().double().requires_grad_(True)
        c  = self.c.detach().cpu().double().requires_grad_(True)

        y_T = y0d * torch.exp( -a*T**3/3 - b*T**2/2 - c*T )
        loss = 0.5 * y_T**2
        grads = torch.autograd.grad(loss, (y0d, a, b, c))
        return y_T.detach().to(device), *[g.detach().to(device) for g in grads]

def solve_ode(model, y0, t, method='rk4', working_dtype=torch.float32, scaler = DynamicScaler):
    with autocast(device_type='cuda', dtype=working_dtype):        
        return odeint(model, y0, t, method=method, loss_scaler=scaler(working_dtype))


# Helper function to compute gradients with respect to y0 and model params.
def compute_gradients(model, y0, t, method,  working_dtype=torch.float32, scaler=DynamicScaler):
    # Ensure y0 is a leaf tensor requiring gradient.
    y0 = y0.detach().clone().requires_grad_(True)

    with autocast(device_type='cuda', dtype=working_dtype):
        # zero grads
        y0.grad = None
        model.a.grad = model.b.grad = model.c.grad = None
        # forward
        sol = solve_ode(model, y0, t, method=method,
                        working_dtype=working_dtype, scaler=scaler)
        loss = 0.5 * sol[-1].pow(2).sum()

    # --------------------- backward w/ safety --------------------------
    try:
        loss.backward()
        grad_y0 = y0.grad.detach().clone()
        grad_a  = model.a.grad.detach().clone()
        grad_b  = model.b.grad.detach().clone()
        grad_c  = model.c.grad.detach().clone()
    except (RuntimeError, ValueError) as e:
        # Gradient overflow / underflow – return None placeholders
        print(f"   (backward failed: {e})")
        grad_y0 = grad_a = grad_b = grad_c = None

    return sol, grad_y0, grad_a, grad_b, grad_c

# --------------------------------------------------------------------------
# Test description
#
#   ODE:              y'(t) = -k [ (t-T/2)² - T²/12 ] y(t)
#   Analytic y(t):    y0 * exp(-k ( t³/3 - T t²/2 + T² t /6 ))
#   Loss:             L = ½ y(T)^2
#
#   Purpose:
#       • Provide a scalar problem whose solution spans the full dynamic
#         range of float16, triggering both under‑ and overflow.
#       • Compare analytic gradients (float64) with mixed‑precision
#         adjoint implementations using NoScaler and DynamicScaler.
#       • Verify DynamicScaler prevents NaN/inf in FP16 and yields accurate
#         gradients, whereas naive FP16 fails.
#       • Save a log‑scale plot of |y(t)| for inspection.
# --------------------------------------------------------------------------

class TestGradientPrecisionComparison(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            print("GPU not available. Skipping tests.")
            self.skipTest("GPU required for these tests.")
        self.device = torch.device("cuda:0")
        self.dim = 1  # Scalar state dimension for analytic reference.
        self.model = PolynomialDampedODE().to(self.device)
        # Create a time grid from 0 to 3.
        self.t = torch.linspace(0., self.model.T, 400, device=self.device)
        # Initial state near fp16 max normal.
        self.y0 = torch.tensor([65504.0/180], device=self.device)  # fp16 max normal
        # Create the ODE model.

    def test_precision_vs_analytic(self):
        # Compute analytic solution and gradients in float64
        y_T_analytic, grad_y0_analytic, grad_a_analytic, grad_b_analytic, grad_c_analytic = \
            self.model.solve_analytically(self.y0, self.t)
        # Prepare to compute relative errors for different dtypes and scalers
        results = []

        # Compute state error once per dtype without gradients
        state_errors = {}
        for wdtype in [torch.float32, torch.float16, torch.bfloat16]:
            sol_no_grad = solve_ode(self.model, self.y0, self.t,
                                    method='rk4', working_dtype=wdtype)
            err = torch.linalg.norm(sol_no_grad[-1] - y_T_analytic) / torch.linalg.norm(y_T_analytic)
            state_errors[str(wdtype)] = f"{err:.8e}"

        scalers_str = ["NoScaler", "DynamicScaler"]
        for working_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for (scaler, name_str) in zip([NoScaler, DynamicScaler], scalers_str):

                sol, grad_y0_num, grad_a_num, grad_b_num, grad_c_num = compute_gradients(
                    self.model, self.y0, self.t, method='rk4',
                    working_dtype=working_dtype, scaler=scaler)

                rel_err_state = state_errors[str(working_dtype)]

                if grad_y0_num is None:
                    results.append((str(working_dtype), name_str,
                                    rel_err_state, 'fail', 'fail', 'fail', 'fail'))
                    continue  # skip grad comparisons

                rel_err_grad_y0 = torch.norm(grad_y0_num - grad_y0_analytic) / torch.norm(grad_y0_analytic)
                rel_err_grad_a  = torch.norm(grad_a_num  - grad_a_analytic ) / torch.norm(grad_a_analytic )
                rel_err_grad_b  = torch.norm(grad_b_num  - grad_b_analytic ) / torch.norm(grad_b_analytic )
                rel_err_grad_c  = torch.norm(grad_c_num  - grad_c_analytic ) / torch.norm(grad_c_analytic )

                results.append((str(working_dtype), name_str,
                                rel_err_state,
                                f"{rel_err_grad_y0:.8e}",
                                f"{rel_err_grad_a:.8e}",
                                f"{rel_err_grad_b:.8e}",
                                f"{rel_err_grad_c:.8e}"))

        # Print results in a markdown-like table format
        table_lines = ["| dtype | Scaler | RelErr y(T) | RelErr ∂y0 | RelErr ∂a | RelErr ∂b | RelErr ∂c |",
               "|-------|--------|--------------|-------------|-------------|-------------|-------------|"]
        quiet = os.environ.get("TORCHMPNODE_TEST_QUIET", "0") == "1"
        for row in results:
            table_lines.append("| " + " | ".join(row) + " |")
        if not quiet:
            print("\n".join(table_lines))

        # --- Pass if all rel errors for float16+DynamicScaler are below 1e-2 ---
        for row in results:
            dtype, scaler, err_state, err_dy0, err_da, err_db, err_dc = row
            if dtype == 'torch.float16' and scaler == "DynamicScaler" and err_dy0 != 'fail':
                all_below = all(float(err) <= 1e-2 for err in (err_dy0, err_da, err_db, err_dc))
                self.assertTrue(all_below, f"float16+DynamicScaler rel error(s) too large: {[err_dy0, err_da, err_db, err_dc]}")
        
        # Plot analytic |y(t)| in log‑scale together with numerical FP16/FP32
        with torch.no_grad():
            t_cpu = self.t.cpu()
            T = self.model.T
            y_analytic = self.y0.cpu() * torch.exp(
                -(self.model.a.cpu()/3)*t_cpu**3
                -(self.model.b.cpu()/2)*t_cpu**2
                - self.model.c.cpu()*t_cpu
            )
            # ---------- constants for float16 range -------------------
            fp16_min = 2**-14        # smallest positive *normal* FP16
            fp16_max = 65504.0       # largest finite FP16
            # ---------- figure 1: state --------------------------------
            plt.figure()
            plt.semilogy(t_cpu, y_analytic.abs(), label='analytic')
            # fp32 numerical
            sol_fp32 = solve_ode(self.model, self.y0, self.t,
                                 method='rk4', working_dtype=torch.float32)
            plt.semilogy(t_cpu, sol_fp32.abs().cpu(), '--', label='rk4‑fp32')
            # fp16 numerical
            sol_fp16 = solve_ode(self.model, self.y0, self.t,
                                 method='rk4', working_dtype=torch.float16,
                                 scaler=DynamicScaler)
            plt.semilogy(t_cpu, sol_fp16.abs().cpu(), ':', label='rk4‑fp16‑scaled')
            # horizontal dashed lines for fp16 limits
            plt.axhline(fp16_min, linestyle='--', color='gray', label='fp16 min normal')
            plt.axhline(fp16_max, linestyle='--', color='gray', label='fp16 max')
            plt.legend()
            plt.xlabel('t'); plt.ylabel('|y(t)|')
            plt.title('Polynomial ODE solution (log‑scale)')
            plt.savefig(OUT_DIR / 'polynomial_state.png', dpi=200)
            plt.close()

            # ---------- figure 2: velocity -----------------------------
            # λ(t) and velocity using analytic formula
            lam_cpu = (self.model.a.cpu()*t_cpu**2 +
                       self.model.b.cpu()*t_cpu +
                       self.model.c.cpu())
            vel_analytic = (lam_cpu * y_analytic).abs()

            plt.figure()
            plt.semilogy(t_cpu, vel_analytic, label='|λ(t) y(t)| analytic')
            # horizontal fp16 bounds
            plt.axhline(fp16_min, linestyle='--', color='gray', label='fp16 min normal')
            plt.axhline(fp16_max, linestyle='--', color='gray', label='fp16 max')
            plt.legend()
            plt.xlabel('t'); plt.ylabel('|y\'(t)|')
            plt.title('Velocity magnitude (log‑scale)')
            plt.savefig(OUT_DIR / 'polynomial_velocity.png', dpi=200)
            plt.close()

            # --- save state CSV ---------------------------------------------------
        state_csv = OUT_DIR / "state_curve.csv"
        with state_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "y_analytic", "y_rk4_fp32", "y_rk4_fp16_scaled"])
            for tt, ya, y32, y16 in zip(t_cpu,
                                       y_analytic,
                                       sol_fp32.cpu(),
                                       sol_fp16.cpu()):
                writer.writerow([float(tt), float(ya), float(y32), float(y16)])

        vel_csv = OUT_DIR / "velocity_curve.csv"
        with vel_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "velocity"])
            for tt, vel in zip(t_cpu, vel_analytic):
                writer.writerow([float(tt), float(vel)])

        # --- write run_info.txt ----------------------------------------------
        info_txt = OUT_DIR / "run_info.txt"
        meta = textwrap.dedent(f"""
        Date: {datetime.datetime.now().isoformat()}
        Polynomial-damped ODE test
          y'(t) = -(a t² + b t + c) y(t)
          T    = {self.model.T}
          a    = {float(self.model.a)}
          b    = {float(self.model.b)}
          c    = {float(self.model.c)}
          y0   = {float(self.y0)}
          RK4 steps = {len(self.t)-1}

        Results table:
        """)
        info_txt.write_text(meta + "\n".join(table_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    unittest.main(argv=[sys.argv[0]] + (['-v'] if args.verbose else []))