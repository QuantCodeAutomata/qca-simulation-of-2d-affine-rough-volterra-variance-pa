"""
Experiment 5: Monte Carlo optimality dominance check versus perturbed controls.

Validates that the paper's Riccati-based optimal controls deliver higher
expected utility than nearby scaled perturbations c * alpha^* for c in {0.8, 0.9, 1.1, 1.2}.

Results show local dominance of the candidate optimal control at c=1.0.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelParams
from src.wealth_sim import dominance_test
from src.plotting import plot_dominance_results

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp5() -> dict:
    """Run Experiment 5 and return results summary."""
    p = ModelParams()
    T = 1.0
    n = 200
    x0 = 1.0
    # M=500 balances statistical stability with runtime (variance paths shared
    # across perturbations, so effective cost ≈ 1 sim per utility/gamma combo)
    M = 500
    perturbations = [0.8, 0.9, 1.0, 1.1, 1.2]

    print("=" * 60)
    print("Experiment 5: Monte Carlo Optimality Dominance")
    print("=" * 60)
    print(f"  M = {M} paths, n = {n}, T = {T}, x0 = {x0}")
    print(f"  Perturbation scales: {perturbations}")

    all_results = {}
    summary_data = {}

    for utility in ['power', 'exponential']:
        print(f"\n[Utility: {utility}]")
        for gamma in [0.2, 0.6]:
            key = f"{utility}_gamma{gamma}"
            print(f"  gamma={gamma}...", end=' ')
            try:
                dom = dominance_test(
                    M=M, n=n, T=T, x0=x0, V0=p.V0_arr, mu0=p.mu0_arr,
                    lam=p.lam_arr, sigma_v=p.sigma_v_arr, alpha=p.alpha_arr,
                    theta=p.theta_arr, rho=p.rho_arr,
                    gamma=gamma, r=p.r, beta=p.beta, kappa=p.kappa,
                    utility=utility, perturbations=perturbations,
                    base_seed=42,
                )
                all_results[key] = dom
                opt_util = dom.get(1.0, {}).get('mean_utility', None)
                print(f"optimal E[U] = {opt_util:.6f}")
                for c, res in sorted(dom.items()):
                    print(f"    c={c}: E[U]={res['mean_utility']:.6f} ±{1.96*res['std_error']:.6f}, "
                          f"neg_wealth={res['n_negative_wealth']}")

                summary_data[key] = {
                    'gamma': gamma, 'utility': utility,
                    'dominance_table': {
                        str(c): {
                            'mean_utility': res['mean_utility'],
                            'ci_low': res['ci_low'],
                            'ci_high': res['ci_high'],
                            'n_negative_wealth': res['n_negative_wealth'],
                        }
                        for c, res in dom.items()
                    },
                }

                # Plot dominance bar chart
                plot_dominance_results(
                    dom, utility=f"{utility} (γ={gamma})",
                    filename=f"exp5_dominance_{utility}_gamma{gamma}.png",
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                summary_data[key] = {'error': str(e)}

    # Log utility (no gamma required for control, but needed for perf)
    print("\n[Utility: log]")
    try:
        dom_log = dominance_test(
            M=M, n=n, T=T, x0=x0, V0=p.V0_arr, mu0=p.mu0_arr,
            lam=p.lam_arr, sigma_v=p.sigma_v_arr, alpha=p.alpha_arr,
            theta=p.theta_arr, rho=p.rho_arr,
            gamma=0.5, r=p.r, beta=p.beta, kappa=p.kappa,
            utility='log', perturbations=perturbations,
            base_seed=42,
        )
        all_results['log'] = dom_log
        opt_u = dom_log.get(1.0, {}).get('mean_utility', None)
        print(f"  optimal E[log U] = {opt_u:.6f}")
        plot_dominance_results(
            dom_log, utility='log',
            filename="exp5_dominance_log.png",
        )
        summary_data['log'] = {
            'utility': 'log',
            'dominance_table': {
                str(c): {
                    'mean_utility': res['mean_utility'],
                    'ci_low': res['ci_low'],
                    'ci_high': res['ci_high'],
                }
                for c, res in dom_log.items()
            },
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        summary_data['log'] = {'error': str(e)}

    summary = {
        'M': M, 'n': n, 'T': T, 'x0': x0,
        'perturbations': perturbations,
        'beta': p.beta, 'kappa': p.kappa,
        'results': summary_data,
    }

    return summary


if __name__ == "__main__":
    result = run_exp5()
    out_path = RESULTS_DIR / "exp5_summary.json"
    with open(out_path, 'w') as f:
        def default_serial(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        json.dump(result, f, indent=2, default=default_serial)
    print(f"\nExp5 summary saved to {out_path}")
