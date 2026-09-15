"""
Experiment 4: Semi-closed-form value function evaluation and formula consistency.

Evaluates V^{exp}, V^{pow}, V^{log} for the baseline parameter set and
performs sensitivity analysis over the missing parameters beta and kappa.

Notes on missing parameters (implementation assumptions documented):
  - zeta = (0,0) for no-liability case
  - x_0 = 1.0 for scale normalization
  - beta and kappa tested over a grid
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelParams
from src.value_functions import (
    evaluate_exponential_value,
    evaluate_power_value,
    evaluate_log_value,
    sensitivity_analysis,
)
from src.plotting import plot_value_sensitivity

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp4() -> dict:
    """Run Experiment 4 and return results summary."""
    p = ModelParams()
    T = 1.0
    n = 200
    x0 = 1.0
    gamma = 0.2

    print("=" * 60)
    print("Experiment 4: Semi-Closed-Form Value Function Evaluation")
    print("=" * 60)
    print(f"  x0 = {x0} (user-supplied baseline, NOT in paper)")
    print(f"  T = {T}, n = {n}, gamma = {gamma}")
    print(f"  beta = {p.beta}, kappa = {p.kappa}")
    print(f"  zeta = {p.zeta} (user-supplied)")

    # ------------------------------------------------------------------ #
    # 1. Baseline value function evaluations
    # ------------------------------------------------------------------ #
    print("\n[1/4] Evaluating baseline value functions...")

    print("  Exponential utility...")
    val_exp = evaluate_exponential_value(
        x0=x0, T=T, n=n, V0=p.V0_arr, mu0=p.mu0_arr,
        alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
        sigma_v=p.sigma_v_arr, lam=p.lam_arr,
        gamma=gamma, r=p.r, beta=p.beta, kappa=p.kappa, zeta=p.zeta_arr,
    )
    print(f"    V^exp = {val_exp['value']:.6f}, exponent = {val_exp['exponent']:.6f}")

    print("  Power utility...")
    val_pow = evaluate_power_value(
        x0=x0, T=T, n=n, V0=p.V0_arr, mu0=p.mu0_arr,
        alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
        sigma_v=p.sigma_v_arr, lam=p.lam_arr,
        gamma=gamma, r=p.r, beta=p.beta, kappa=p.kappa,
    )
    print(f"    V^pow = {val_pow['value']:.6f}, exponent = {val_pow['exponent']:.6f}")

    print("  Log utility...")
    val_log = evaluate_log_value(
        x0=x0, T=T, n=n, V0=p.V0_arr, mu0=p.mu0_arr,
        alpha=p.alpha_arr, theta=p.theta_arr, lam=p.lam_arr, r=p.r,
    )
    print(f"    V^log = {val_log['value']:.6f}")

    # ------------------------------------------------------------------ #
    # 2. Finiteness and quadrature refinement checks
    # ------------------------------------------------------------------ #
    print("\n[2/4] Quadrature refinement check (power utility)...")
    refinement_vals = {}
    for n_ref in [200, 400, 800]:
        vr = evaluate_power_value(
            x0=x0, T=T, n=n_ref, V0=p.V0_arr, mu0=p.mu0_arr,
            alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
            sigma_v=p.sigma_v_arr, lam=p.lam_arr,
            gamma=gamma, r=p.r, beta=p.beta, kappa=p.kappa,
        )
        refinement_vals[n_ref] = float(vr['value'])
        print(f"  n={n_ref}: V^pow = {vr['value']:.8f}")

    # ------------------------------------------------------------------ #
    # 3. Sensitivity over beta, kappa
    # ------------------------------------------------------------------ #
    print("\n[3/4] Sensitivity analysis over (beta, kappa)...")
    beta_grid = [0.0, 0.5, 1.0]
    kappa_grid = [0.0, 0.1, 0.5]

    sens_pow = sensitivity_analysis(
        x0=x0, T=T, n=n, V0=p.V0_arr, mu0=p.mu0_arr,
        alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
        sigma_v=p.sigma_v_arr, lam=p.lam_arr,
        gamma=gamma, r=p.r, beta_grid=beta_grid, kappa_grid=kappa_grid,
        utility='power',
    )
    sens_exp = sensitivity_analysis(
        x0=x0, T=T, n=n, V0=p.V0_arr, mu0=p.mu0_arr,
        alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
        sigma_v=p.sigma_v_arr, lam=p.lam_arr,
        gamma=gamma, r=p.r, beta_grid=beta_grid, kappa_grid=kappa_grid,
        utility='exponential',
    )
    print("  Power utility sensitivity (beta x kappa):")
    for k, v in sens_pow.items():
        print(f"    beta={k[0]}, kappa={k[1]}: {v:.6f}" if isinstance(v, float) else f"    beta={k[0]}, kappa={k[1]}: {v}")

    # Plot sensitivity heatmap
    print("\n[4/4] Generating sensitivity heatmap...")
    plot_value_sensitivity(
        sens_pow, beta_grid, kappa_grid, utility='power',
        filename="exp4_sensitivity_power.png",
    )
    plot_value_sensitivity(
        sens_exp, beta_grid, kappa_grid, utility='exponential',
        filename="exp4_sensitivity_exponential.png",
    )

    # ------------------------------------------------------------------ #
    # 4. Zeta sensitivity (exponential utility)
    # ------------------------------------------------------------------ #
    print("  Zeta sensitivity (exponential)...")
    zeta_vals = {}
    for zeta_val in [0.0, 0.01, 0.05]:
        zeta_t = np.full(2, zeta_val)
        vz = evaluate_exponential_value(
            x0=x0, T=T, n=n, V0=p.V0_arr, mu0=p.mu0_arr,
            alpha=p.alpha_arr, theta=p.theta_arr, rho=p.rho_arr,
            sigma_v=p.sigma_v_arr, lam=p.lam_arr,
            gamma=gamma, r=p.r, beta=p.beta, kappa=p.kappa, zeta=zeta_t,
        )
        zeta_vals[zeta_val] = float(vz['value'])
        print(f"    zeta={zeta_val}: V^exp = {vz['value']:.6f}")

    def ser_key(k):
        if isinstance(k, tuple):
            return str(k)
        return k

    summary = {
        'x0': x0, 'T': T, 'n': n, 'gamma': gamma,
        'baseline': {
            'exponential': val_exp,
            'power': val_pow,
            'log': val_log,
        },
        'quadrature_refinement_power': refinement_vals,
        'sensitivity_power': {str(k): float(v) for k, v in sens_pow.items() if isinstance(v, float)},
        'sensitivity_exponential': {str(k): float(v) for k, v in sens_exp.items() if isinstance(v, float)},
        'zeta_sensitivity_exponential': zeta_vals,
    }

    return summary


if __name__ == "__main__":
    result = run_exp4()
    out_path = RESULTS_DIR / "exp4_summary.json"
    with open(out_path, 'w') as f:
        def default_serial(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        json.dump(result, f, indent=2, default=default_serial)
    print(f"\nExp4 summary saved to {out_path}")
