"""
Experiment 3: Deterministic optimal strategy curves across utility, horizon,
and risk aversion (Figure 3-type).

Computes pi_{i,t}^* for:
  - Horizons T in {1, 5, 10}
  - Risk aversion gamma in {0.2, 0.6, 0.8}
  - Utilities: exponential, power

Grid choice: n = 200 for all horizons (consistent with Figures 1 and 2).
This choice is explicitly stated because the paper does not specify n for Figure 3.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelParams
from src.strategy import compute_strategy_curves
from src.plotting import plot_strategy_curves

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp3() -> dict:
    """Run Experiment 3 and return results summary."""
    p = ModelParams()
    n = 200
    T_values = [1, 5, 10]
    gamma_values = [0.2, 0.6, 0.8]
    utilities = ['exponential', 'power']

    print("=" * 60)
    print("Experiment 3: Optimal Strategy Curves")
    print("=" * 60)
    print(f"  Horizons: {T_values}, Gammas: {gamma_values}")
    print(f"  Grid: n={n} per horizon (stated assumption)")
    print(f"  zeta = {p.zeta} (user-supplied default)")

    all_results = {}
    summary_data = {}

    for utility in utilities:
        all_results[utility] = {}
        summary_data[utility] = {}
        print(f"\n  Utility: {utility}")

        for T in T_values:
            all_results[utility][T] = {}
            summary_data[utility][T] = {}

            for gamma in gamma_values:
                print(f"    T={T}, gamma={gamma}...", end=' ')
                tg, pi, psi = compute_strategy_curves(
                    n=n, T=float(T), alpha=p.alpha_arr, theta=p.theta_arr,
                    rho=p.rho_arr, sigma_v=p.sigma_v_arr, lam=p.lam_arr,
                    gamma=gamma, r=p.r, utility=utility, zeta=p.zeta_arr,
                )
                all_results[utility][T][gamma] = (tg, pi, psi)
                summary_data[utility][T][gamma] = {
                    'pi_start': pi[0].tolist(),
                    'pi_end': pi[-1].tolist(),
                    'pi_min': pi.min(axis=0).tolist(),
                    'pi_max': pi.max(axis=0).tolist(),
                }
                print(f"pi[0]={pi[0]}, pi[-1]={pi[-1]}")

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    print("\n  Generating strategy plots...")
    plot_strategy_curves(all_results, filename="exp3_strategy_curves.png")

    # ------------------------------------------------------------------ #
    # Zeta sensitivity for exponential utility
    # ------------------------------------------------------------------ #
    print("  Running zeta sensitivity (exponential)...")
    T_test, gamma_test = 1, 0.2
    zeta_sensitivity = {}
    for zeta_val in [0.0, 0.01]:
        zeta_test = np.full(2, zeta_val)
        tg_z, pi_z, _ = compute_strategy_curves(
            n=n, T=float(T_test), alpha=p.alpha_arr, theta=p.theta_arr,
            rho=p.rho_arr, sigma_v=p.sigma_v_arr, lam=p.lam_arr,
            gamma=gamma_test, r=p.r, utility='exponential', zeta=zeta_test,
        )
        zeta_sensitivity[zeta_val] = {
            'pi_start': pi_z[0].tolist(),
            'pi_end': pi_z[-1].tolist(),
        }
        print(f"    zeta={zeta_val}: pi[0]={pi_z[0]}")

    summary = {
        'n_per_horizon': n,
        'T_values': T_values,
        'gamma_values': gamma_values,
        'grid_choice_note': 'n=200 per horizon (paper does not specify n for Figure 3)',
        'zeta_assumption': p.zeta,
        'strategy_summary': summary_data,
        'zeta_sensitivity_exponential': zeta_sensitivity,
    }

    return summary


if __name__ == "__main__":
    result = run_exp3()
    out_path = RESULTS_DIR / "exp3_summary.json"

    def default_serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=default_serial)
    print(f"\nExp3 summary saved to {out_path}")
