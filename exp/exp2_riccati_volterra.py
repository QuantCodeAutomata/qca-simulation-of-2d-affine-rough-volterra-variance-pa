"""
Experiment 2: Numerical solution of utility-specific Riccati-Volterra equations.

Solves the fractional Adams-Bashforth-Moulton system for exponential and
power utility (paper equations 32-33 and 50-51) and reproduces Figure 2-type
curves for psi^1(t), psi^2(t) over [0, 1].

Key findings validated:
  (1) psi is non-positive for exponential utility.
  (2) psi is positive for power utility.
  (3) Grid refinement confirms numerical stability.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelParams
from src.riccati import solve_riccati_volterra
from src.plotting import plot_riccati_solutions

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def solve_and_convert_to_time(psi_tau: np.ndarray, T: float) -> tuple:
    """
    Convert psi from tau-forward indexing to t-forward indexing.

    psi_tau[k] = psi(tau_k) where tau_k = k*Delta.
    Time t = T - tau => psi(T-t_j) = psi_tau[n-j].

    Returns psi_t: shape (n+1, d) indexed forward in t.
    """
    n = psi_tau.shape[0] - 1
    Delta = T / n
    t_grid = np.arange(n + 1) * Delta
    # psi at time t_j: need psi(T - t_j) = psi_tau[n - j]
    psi_t = psi_tau[::-1, :]   # reverse: psi_t[j] = psi(T - t_j)... wait
    # Actually psi_tau[k] at tau=k*Delta = T - t => t = T - k*Delta
    # For plotting against t: at t = j*Delta, tau = T - j*Delta = (n-j)*Delta
    # So psi at time t_j uses psi_tau[n-j]
    # psi_t[j] = psi_tau[n - j]  for j = 0, ..., n
    psi_t = psi_tau[np.arange(n + 1)[::-1], :]  # equivalent to [::-1]
    return t_grid, psi_t


def run_exp2() -> dict:
    """Run Experiment 2 and return results summary."""
    p = ModelParams()
    T = 1.0
    gamma = 0.2
    n = 200

    print("=" * 60)
    print("Experiment 2: Riccati-Volterra Equation Solutions")
    print("=" * 60)
    print(f"  gamma = {gamma}, n = {n}, T = {T}")
    print(f"  zeta = {p.zeta} (user-supplied default, NOT in paper)")

    # ------------------------------------------------------------------ #
    # 1. Solve exponential utility Riccati
    # ------------------------------------------------------------------ #
    print("\n[1/4] Solving exponential utility Riccati system...")
    psi_exp_tau = solve_riccati_volterra(
        n=n, T=T, alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
        sigma_v=p.sigma_v_arr, lam=p.lam_arr,
        gamma=gamma, utility='exponential', zeta=p.zeta_arr,
    )
    t_grid_exp, psi_exp_t = solve_and_convert_to_time(psi_exp_tau, T)

    print(f"  psi_exp range: [{psi_exp_t.min():.6f}, {psi_exp_t.max():.6f}]")
    exp_nonpos = bool(np.all(psi_exp_t <= 1e-10))
    print(f"  All non-positive: {exp_nonpos}")

    # ------------------------------------------------------------------ #
    # 2. Solve power utility Riccati
    # ------------------------------------------------------------------ #
    print("[2/4] Solving power utility Riccati system...")
    psi_pow_tau = solve_riccati_volterra(
        n=n, T=T, alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
        sigma_v=p.sigma_v_arr, lam=p.lam_arr,
        gamma=gamma, utility='power',
    )
    t_grid_pow, psi_pow_t = solve_and_convert_to_time(psi_pow_tau, T)

    print(f"  psi_pow range: [{psi_pow_t.min():.6f}, {psi_pow_t.max():.6f}]")
    pow_pos = bool(np.all(psi_pow_t >= -1e-10))
    print(f"  All non-negative: {pow_pos}")

    # ------------------------------------------------------------------ #
    # 3. Plot (Figure 2-type)
    # ------------------------------------------------------------------ #
    print("[3/4] Generating Figure 2-type plots...")
    plot_riccati_solutions(
        psi_exp=psi_exp_t,
        psi_pow=psi_pow_t,
        time_grid=t_grid_exp,
        gamma=gamma,
    )

    # ------------------------------------------------------------------ #
    # 4. Grid refinement / convergence check
    # ------------------------------------------------------------------ #
    print("[4/4] Running grid refinement check...")
    refinement_results = {}
    for n_ref in [200, 400, 800]:
        psi_ref = solve_riccati_volterra(
            n=n_ref, T=T, alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
            sigma_v=p.sigma_v_arr, lam=p.lam_arr,
            gamma=gamma, utility='power',
        )
        _, psi_ref_t = solve_and_convert_to_time(psi_ref, T)
        refinement_results[n_ref] = {
            'psi_min': float(psi_ref_t.min()),
            'psi_max': float(psi_ref_t.max()),
            'all_nonneg': bool(np.all(psi_ref_t >= -1e-10)),
        }
        print(f"  n={n_ref}: min={psi_ref_t.min():.6f}, max={psi_ref_t.max():.6f}")

    # ------------------------------------------------------------------ #
    # 5. Zeta sensitivity for exponential utility
    # ------------------------------------------------------------------ #
    print("  Running zeta sensitivity check...")
    zeta_results = {}
    for zeta_val in [0.0, 0.01, -0.01]:
        zeta_test = np.full(2, zeta_val)
        psi_zeta = solve_riccati_volterra(
            n=n, T=T, alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
            sigma_v=p.sigma_v_arr, lam=p.lam_arr,
            gamma=gamma, utility='exponential', zeta=zeta_test,
        )
        _, psi_z_t = solve_and_convert_to_time(psi_zeta, T)
        zeta_results[zeta_val] = {
            'psi_min': float(psi_z_t.min()),
            'psi_max': float(psi_z_t.max()),
            'all_nonpos': bool(np.all(psi_z_t <= 1e-8)),
        }
        print(f"  zeta={zeta_val}: min={psi_z_t.min():.6f}, max={psi_z_t.max():.6f}")

    summary = {
        'n': n, 'T': T, 'gamma': gamma,
        'zeta_assumption': p.zeta,
        'exponential': {
            'psi_min': float(psi_exp_t.min()),
            'psi_max': float(psi_exp_t.max()),
            'all_nonpositive': exp_nonpos,
        },
        'power': {
            'psi_min': float(psi_pow_t.min()),
            'psi_max': float(psi_pow_t.max()),
            'all_nonnegative': pow_pos,
        },
        'grid_refinement_power': refinement_results,
        'zeta_sensitivity_exponential': zeta_results,
    }

    return summary


if __name__ == "__main__":
    result = run_exp2()
    out_path = RESULTS_DIR / "exp2_summary.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x)
    print(f"\nExp2 summary saved to {out_path}")
