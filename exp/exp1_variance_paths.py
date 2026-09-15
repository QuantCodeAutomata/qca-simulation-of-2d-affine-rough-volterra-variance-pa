"""
Experiment 1: Simulation of 2D affine rough-Volterra variance paths with upward jumps.

Reproduces Figure 1-type outputs: five sample paths for each variance component
over a one-year horizon on a uniform grid with 200 steps.

Key findings validated:
  (1) Visible upward spikes in both components when jumps are enabled.
  (2) Rougher-looking paths for alpha_1=0.6 (H=0.1) vs alpha_2=0.9 (H=0.4).
  (3) Post-spike mean-reverting relaxation shaped by the Volterra kernel.
  (4) Jump vs no-jump comparison confirms jump-induced spikes.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelParams
from src.volterra_sim import simulate_variance_paths, compute_roughness_diagnostics, check_positivity
from src.plotting import plot_variance_paths, plot_jump_comparison, plot_roughness_comparison

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1() -> dict:
    """Run Experiment 1 and return a results summary."""
    p = ModelParams()
    T = 1.0
    n = 200
    n_paths = 5

    print("=" * 60)
    print("Experiment 1: 2D Rough-Volterra Variance Path Simulation")
    print("=" * 60)
    print(f"  alpha = {p.alpha}  =>  H = {p.hurst}")
    print(f"  V0 = {p.V0}")
    print(f"  beta = {p.beta} (user-supplied, NOT in paper)")
    print(f"  kappa = {p.kappa} (user-supplied, NOT in paper)")

    # ------------------------------------------------------------------ #
    # 1. Baseline: with jumps (beta=0.5, kappa=0.3)
    # ------------------------------------------------------------------ #
    print("\n[1/4] Simulating with jumps...")
    V_jump, time_grid = simulate_variance_paths(
        n=n, T=T, V0=p.V0_arr, mu0=p.mu0_arr, lam=p.lam_arr,
        sigma_v=p.sigma_v_arr, alpha=p.alpha_arr,
        beta=p.beta, kappa=p.kappa,
        n_paths=n_paths, enable_jumps=True, rng_seed=42,
    )
    plot_variance_paths(
        V_jump, time_grid,
        title_suffix=f"With Jumps (β={p.beta}, κ={p.kappa})",
        filename="exp1_variance_paths_with_jumps.png",
        alpha_params=p.alpha,
    )

    # ------------------------------------------------------------------ #
    # 2. No-jump benchmark (beta=0)
    # ------------------------------------------------------------------ #
    print("[2/4] Simulating no-jump benchmark...")
    V_nojump, _ = simulate_variance_paths(
        n=n, T=T, V0=p.V0_arr, mu0=p.mu0_arr, lam=p.lam_arr,
        sigma_v=p.sigma_v_arr, alpha=p.alpha_arr,
        beta=0.0, kappa=p.kappa,
        n_paths=n_paths, enable_jumps=False, rng_seed=42,
    )
    plot_variance_paths(
        V_nojump, time_grid,
        title_suffix="No Jumps (β=0)",
        filename="exp1_variance_paths_no_jumps.png",
        alpha_params=p.alpha,
    )

    # Jump vs no-jump comparison plot
    plot_jump_comparison(V_jump, V_nojump, time_grid, alpha_params=p.alpha)

    # ------------------------------------------------------------------ #
    # 3. Roughness comparison: swap alpha values
    # ------------------------------------------------------------------ #
    print("[3/4] Simulating roughness comparison...")
    alpha_swapped = np.array([p.alpha[1], p.alpha[0]])   # [0.9, 0.6] instead of [0.6, 0.9]
    V_swapped, _ = simulate_variance_paths(
        n=n, T=T, V0=p.V0_arr, mu0=p.mu0_arr, lam=p.lam_arr,
        sigma_v=p.sigma_v_arr, alpha=alpha_swapped,
        beta=p.beta, kappa=p.kappa,
        n_paths=n_paths, enable_jumps=True, rng_seed=42,
    )
    plot_roughness_comparison(
        V_jump, V_swapped, time_grid,
        alpha_rough=p.alpha,
        alpha_smooth=alpha_swapped.tolist(),
    )

    # ------------------------------------------------------------------ #
    # 4. Diagnostics
    # ------------------------------------------------------------------ #
    print("[4/4] Computing roughness diagnostics...")
    roughness = compute_roughness_diagnostics(V_jump, time_grid)
    positivity = check_positivity(V_jump)

    # Hurst surrogates should show comp1 rougher than comp2
    h1 = roughness.get('component_1', {}).get('hurst_surrogate', np.nan)
    h2 = roughness.get('component_2', {}).get('hurst_surrogate', np.nan)
    print(f"  Hurst surrogate comp1 = {h1:.4f}, comp2 = {h2:.4f}")
    print(f"  Expected: H1 < H2 (rougher comp1)")
    print(f"  Positivity: {positivity}")

    summary = {
        'n': n, 'T': T, 'n_paths': n_paths,
        'alpha': p.alpha, 'hurst': p.hurst,
        'beta': p.beta, 'kappa': p.kappa,
        'roughness_diagnostics': roughness,
        'positivity_report': positivity,
        'hurst_surrogate': {'component_1': h1, 'component_2': h2},
        'qualitative_rougher_comp1': h1 < h2 if not np.isnan(h1 + h2) else None,
    }

    return summary


if __name__ == "__main__":
    result = run_exp1()
    out_path = RESULTS_DIR / "exp1_summary.json"
    with open(out_path, 'w') as f:
        # Make JSON-serializable
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        import json
        json.dump(result, f, indent=2, default=to_serializable)
    print(f"\nExp1 summary saved to {out_path}")
